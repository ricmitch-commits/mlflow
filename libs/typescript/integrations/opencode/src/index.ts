/**
 * MLflow Tracing Plugin for OpenCode.
 *
 * This plugin listens for session completion events and creates MLflow traces
 * directly using the @mlflow/core TypeScript SDK.
 *
 * Usage:
 *   1. Install: npm install @mlflow/opencode @mlflow/core
 *   2. Add to opencode.json:
 *      { "$schema": "https://opencode.ai/config.json", "plugins": ["@mlflow/opencode"] }
 *   3. Set environment variables:
 *      export MLFLOW_TRACKING_URI=http://localhost:5000
 *      export MLFLOW_EXPERIMENT_ID=123
 *      # For a UC-backed experiment: export MLFLOW_TRACE_LOCATION=catalog.schema.table_prefix
 *   4. Run opencode normally - tracing happens automatically
 */

import { Plugin } from '@opencode/plugin';
import {
  init,
  startSpan,
  withSpan,
  updateCurrentTrace,
  flushTraces,
  SpanType,
  SpanAttributeKey,
} from '@mlflow/core';

// Avoid routine console output so the plugin does not interrupt the TUI.
const DEBUG = process.env.MLFLOW_OPENCODE_DEBUG === 'true';

const NANOSECONDS_PER_MS = 1e6;
const MAX_PREVIEW_LENGTH = 1000;
const MESSAGE_ROLE_USER = 'user';
const MESSAGE_ROLE_ASSISTANT = 'assistant';
const PART_TYPE_TEXT = 'text';
const PART_TYPE_TOOL = 'tool';
const PART_TYPE_REASONING = 'reasoning';
const MAX_TRANSCRIPT_MESSAGES = 2000;
const MAX_CONCURRENT_SESSIONS = 4;
const SUBSCRIPTION_RETRY_INITIAL_MS = 100;
const SUBSCRIPTION_RETRY_MAX_MS = 5000;
const CLEANUP_TIMEOUT_MS = 1000;

type SessionOutcomeStatus = 'succeeded' | 'failed' | 'interrupted';

interface SessionOutcome {
  status: SessionOutcomeStatus;
  reason?: string;
  error?: string;
}

// Well-known trace metadata keys. We pass these through updateCurrentTrace's
// generic `metadata` option rather than its `sessionId`/`user` convenience
// options, since those options (and the TraceMetadataKey.TRACE_SESSION/TRACE_USER
// enum members) are only present in newer @mlflow/core builds. The metadata keys
// themselves are stable, so this works across the published version range.
const TRACE_SESSION_METADATA_KEY = 'mlflow.trace.session';
const TRACE_USER_METADATA_KEY = 'mlflow.trace.user';

function getCurrentUser(): string {
  return process.env.USER || process.env.USERNAME || '';
}

let initialized = false;
let lastInitializationWarning: string | null = null;

interface JsonObject {
  [key: string]: unknown;
}

interface MessagePart {
  type: string;
  text?: string;
  tool?: string;
  callID?: string;
  state?: {
    status?: string;
    input?: JsonObject;
    output?: string;
    error?: string;
    title?: string;
    time?: {
      start?: number;
      end?: number;
    };
  };
}

interface MessageInfo {
  role?: string;
  modelID?: string;
  providerID?: string;
  finish?: string;
  error?: string;
  tokens?: {
    input?: number;
    output?: number;
    reasoning?: number;
    cache?: {
      read?: number;
      write?: number;
    };
  };
  time?: {
    created?: number;
    completed?: number;
  };
}

interface Message {
  id?: string;
  info: MessageInfo;
  parts: MessagePart[];
}

type ToolCall = {
  id: string;
  type: 'function';
  function: { name: string; arguments: string };
};

type ConversationMessage = {
  role: string;
  content: string;
  reasoning?: string;
  tool_call_id?: string;
  tool_calls?: ToolCall[];
};

interface SessionState {
  transcript: Message[];
  processedTurnKey?: string;
  outcome?: SessionOutcome;
  pending: boolean;
  queued: boolean;
  inFlight?: Promise<void>;
}

type PluginDefinition = Parameters<typeof Plugin.define>[0];
type PluginContext = Parameters<PluginDefinition['setup']>[0];

/**
 * Initialize the MLflow tracing SDK if not already initialized.
 * Requires MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_ID environment variables.
 * MLFLOW_TRACE_LOCATION is optional and resolved by the core SDK. If init()
 * fails, a later event retries it; repeated warnings are suppressed.
 */
function ensureInitialized(): boolean {
  if (initialized) {
    return true;
  }

  const trackingUri = process.env.MLFLOW_TRACKING_URI;
  const experimentId = process.env.MLFLOW_EXPERIMENT_ID;

  if (!trackingUri) {
    if (DEBUG) {
      console.error('[mlflow] MLFLOW_TRACKING_URI not set, skipping initialization');
    }
    return false;
  }

  if (!experimentId) {
    if (DEBUG) {
      console.error('[mlflow] MLFLOW_EXPERIMENT_ID not set, skipping initialization');
    }
    return false;
  }

  try {
    init({ trackingUri, experimentId });
    initialized = true;
    lastInitializationWarning = null;
    if (DEBUG) {
      console.error('[mlflow] SDK initialized successfully');
    }
    return true;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (message !== lastInitializationWarning) {
      console.error(`[mlflow] OpenCode tracing is disabled: ${message}`);
      lastInitializationWarning = message;
    }
    return false;
  }
}

function asObject(value: unknown): JsonObject | undefined {
  return typeof value === 'object' && value != null && !Array.isArray(value)
    ? (value as JsonObject)
    : undefined;
}

function asString(value: unknown): string | undefined {
  return typeof value === 'string' ? value : undefined;
}

function asNumber(value: unknown): number | undefined {
  return typeof value === 'number' ? value : undefined;
}

function timestampToNs(timestamp: number | undefined): number | undefined {
  return timestamp != null ? Math.floor(timestamp * NANOSECONDS_PER_MS) : undefined;
}

function normalizeTime(value: unknown): MessageInfo['time'] {
  const time = asObject(value);
  if (!time) {
    return undefined;
  }

  return {
    created: asNumber(time.created),
    completed: asNumber(time.completed),
  };
}

function normalizeTokens(value: unknown): MessageInfo['tokens'] {
  const tokens = asObject(value);
  if (!tokens) {
    return undefined;
  }

  const cache = asObject(tokens.cache);
  return {
    input: asNumber(tokens.input),
    output: asNumber(tokens.output),
    reasoning: asNumber(tokens.reasoning),
    ...(cache
      ? {
          cache: {
            read: asNumber(cache.read),
            write: asNumber(cache.write),
          },
        }
      : {}),
  };
}

function jsonStringify(value: unknown): string {
  try {
    return JSON.stringify(value) ?? String(value);
  } catch {
    return String(value);
  }
}

/** Convert V2 Tool.Content arrays into a useful string for MLflow inputs/outputs. */
function toolContentToText(value: unknown): string {
  if (typeof value === 'string') {
    return value;
  }

  if (Array.isArray(value)) {
    return value
      .map((item) => toolContentToText(item))
      .filter(Boolean)
      .join('\n');
  }

  const content = asObject(value);
  if (!content) {
    return value == null ? '' : jsonStringify(value);
  }

  if (typeof content.text === 'string') {
    return content.text;
  }

  if (content.type === 'file') {
    const name = asString(content.name);
    const uri = asString(content.uri);
    return `[file: ${name || uri || 'unknown'}]`;
  }

  return jsonStringify(value);
}

function errorToText(value: unknown): string | undefined {
  if (typeof value === 'string') {
    return value;
  }

  const error = asObject(value);
  if (!error) {
    return value == null ? undefined : jsonStringify(value);
  }

  return asString(error.message) || jsonStringify(value);
}

function normalizeToolState(value: unknown): MessagePart['state'] {
  const state = asObject(value);
  if (!state) {
    return undefined;
  }

  const input = asObject(state.input);
  const hasContent = state.content !== undefined;
  const output = asString(state.output);
  const error = errorToText(state.error);
  const time = asObject(state.time);

  return {
    status: asString(state.status),
    ...(input ? { input } : {}),
    ...(hasContent ? { output: toolContentToText(state.content) } : output ? { output } : {}),
    ...(error ? { error } : {}),
    ...(asString(state.title) ? { title: asString(state.title) } : {}),
    ...(time
      ? {
          time: {
            start: asNumber(time.start),
            end: asNumber(time.end),
          },
        }
      : {}),
  };
}

function normalizeToolPart(value: unknown): MessagePart | undefined {
  const part = asObject(value);
  if (!part || part.type !== PART_TYPE_TOOL) {
    return undefined;
  }

  const partTime = asObject(part.time);
  const startTime = asNumber(partTime?.ran) ?? asNumber(partTime?.created);
  const endTime = asNumber(partTime?.completed);
  const state = normalizeToolState(part.state) || {};
  const oldTime = asObject(state.time);

  return {
    type: PART_TYPE_TOOL,
    tool: asString(part.name) || asString(part.tool) || 'unknown',
    callID: asString(part.id) || asString(part.callID),
    state: {
      ...state,
      time: {
        start: asNumber(oldTime?.start) ?? startTime,
        end: asNumber(oldTime?.end) ?? endTime,
      },
    },
  };
}

function normalizeAssistantPart(value: unknown): MessagePart | undefined {
  const part = asObject(value);
  if (!part) {
    return undefined;
  }

  if (part.type === PART_TYPE_TEXT || part.type === PART_TYPE_REASONING) {
    return {
      type: part.type,
      text: asString(part.text) || '',
    };
  }

  return normalizeToolPart(value);
}

function normalizeLegacyPart(value: unknown): MessagePart | undefined {
  const part = asObject(value);
  if (!part || typeof part.type !== 'string') {
    return undefined;
  }

  if (part.type === PART_TYPE_TEXT || part.type === PART_TYPE_REASONING) {
    return {
      type: part.type,
      text: asString(part.text) || '',
    };
  }

  return normalizeToolPart(part);
}

function normalizeUserText(message: JsonObject): string {
  const text = asString(message.text);
  const files = Array.isArray(message.files) ? message.files : [];
  const attachments = files.map((value) => {
    const file = asObject(value);
    const name = asString(file?.name);
    const mime = asString(file?.mime);
    const description = asString(file?.description);
    const label = name || mime || 'attachment';
    return description ? `${label}: ${description}` : label;
  });

  const attachmentText = attachments.length > 0 ? `Attached files: ${attachments.join(', ')}` : '';
  if (text?.trim() && attachmentText) {
    return `${text}\n${attachmentText}`;
  }
  return text?.trim() ? text : attachmentText;
}

/**
 * Normalize both V2 session messages and the former V1 shape into the small
 * internal representation used by the tracing code. V2 uses `text` on user
 * messages and `model`/`content` on assistant messages; it no longer has the
 * V1 `info`/`parts` envelope.
 */
function normalizeMessage(value: unknown): Message | undefined {
  const message = asObject(value);
  if (!message) {
    return undefined;
  }

  const messageType = asString(message.type);
  if (messageType === MESSAGE_ROLE_USER) {
    return {
      id: asString(message.id),
      info: {
        role: MESSAGE_ROLE_USER,
        time: normalizeTime(message.time),
      },
      parts: [
        {
          type: PART_TYPE_TEXT,
          text: normalizeUserText(message),
        },
      ],
    };
  }

  if (messageType === MESSAGE_ROLE_ASSISTANT) {
    const model = asObject(message.model);
    const content = Array.isArray(message.content) ? message.content : [];
    return {
      id: asString(message.id),
      info: {
        role: MESSAGE_ROLE_ASSISTANT,
        modelID: asString(model?.id),
        providerID: asString(model?.providerID),
        finish: asString(message.finish),
        error: errorToText(message.error),
        tokens: normalizeTokens(message.tokens),
        time: normalizeTime(message.time),
      },
      parts: content
        .map(normalizeAssistantPart)
        .filter((part): part is MessagePart => part !== undefined),
    };
  }

  // System, synthetic, skill, shell, and compaction records are lifecycle
  // metadata rather than conversational turns, so they are intentionally
  // omitted from the normalized trace transcript. Keep the V1 fallback so
  // local consumers can upgrade OpenCode without an abrupt failure.
  const legacyInfo = asObject(message.info);
  const legacyRole = asString(legacyInfo?.role);
  if (legacyRole !== MESSAGE_ROLE_USER && legacyRole !== MESSAGE_ROLE_ASSISTANT) {
    return undefined;
  }

  const legacyParts = Array.isArray(message.parts) ? message.parts : [];
  return {
    id: asString(message.id),
    info: {
      role: legacyRole,
      modelID: asString(legacyInfo?.modelID),
      providerID: asString(legacyInfo?.providerID),
      tokens: normalizeTokens(legacyInfo?.tokens),
      time: normalizeTime(legacyInfo?.time),
    },
    parts: legacyParts
      .map(normalizeLegacyPart)
      .filter((part): part is MessagePart => part !== undefined),
  };
}

function normalizeMessages(value: unknown): Message[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.map(normalizeMessage).filter((message): message is Message => message !== undefined);
}

/** Merge post-compaction context with the bounded ID-keyed session history. */
function mergeTranscript(previous: Message[], current: Message[]): Message[] {
  if (previous.length === 0 || current.length === 0 || current.some((message) => !message.id)) {
    return current;
  }

  const currentIds = new Set(current.map((message) => message.id));
  const retained = previous.filter((message) => !message.id || !currentIds.has(message.id));
  return [...retained, ...current].slice(-MAX_TRANSCRIPT_MESSAGES);
}

function extractUserPrompt(messages: Message[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.info.role === MESSAGE_ROLE_USER) {
      const textParts = msg.parts
        .filter((part) => part.type === PART_TYPE_TEXT && part.text)
        .map((part) => part.text || '');
      if (textParts.length > 0) {
        return textParts.join('\n');
      }
    }
  }
  return '';
}

function extractAssistantResponse(messages: Message[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.info.role === MESSAGE_ROLE_ASSISTANT) {
      const textParts = msg.parts
        .filter((part) => part.type === PART_TYPE_TEXT && part.text?.trim())
        .map((part) => part.text || '');
      if (textParts.length > 0) {
        return textParts.join('\n');
      }
    }
  }
  return '';
}

function findLastUserMessageIndex(messages: Message[]): number | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].info.role === MESSAGE_ROLE_USER) {
      return i;
    }
  }
  return null;
}

function toolCallFromPart(part: MessagePart): ToolCall {
  return {
    id: part.callID || '',
    type: 'function',
    function: {
      name: part.tool || 'unknown',
      arguments: JSON.stringify(part.state?.input ?? {}),
    },
  };
}

/** Reconstruct conversation history in the OpenAI chat-completion shape. */
function reconstructConversationMessages(
  messages: Message[],
  endIdx: number,
): ConversationMessage[] {
  const result: ConversationMessage[] = [];

  for (let i = 0; i < endIdx; i++) {
    const msg = messages[i];
    const role = msg.info.role;
    const parts = msg.parts;

    if (role === MESSAGE_ROLE_USER) {
      const textParts = parts
        .filter((part) => part.type === PART_TYPE_TEXT && part.text)
        .map((part) => part.text || '');
      if (textParts.length > 0) {
        result.push({ role: MESSAGE_ROLE_USER, content: textParts.join('\n') });
      }
      continue;
    }

    if (role !== MESSAGE_ROLE_ASSISTANT) {
      continue;
    }

    const textParts = parts
      .filter((part) => part.type === PART_TYPE_TEXT && part.text)
      .map((part) => part.text || '');
    const reasoningParts = parts
      .filter((part) => part.type === PART_TYPE_REASONING && part.text)
      .map((part) => part.text || '');
    const toolParts = parts.filter((part) => part.type === PART_TYPE_TOOL);

    if (textParts.length > 0 || reasoningParts.length > 0 || toolParts.length > 0) {
      const assistantMessage: ConversationMessage = {
        role: MESSAGE_ROLE_ASSISTANT,
        content: textParts.join('\n'),
      };
      if (reasoningParts.length > 0) {
        assistantMessage.reasoning = reasoningParts.join('\n\n');
      }
      if (toolParts.length > 0) {
        assistantMessage.tool_calls = toolParts.map(toolCallFromPart);
      }
      result.push(assistantMessage);
    }

    for (const part of toolParts) {
      if (part.state?.status === 'completed' || part.state?.status === 'error') {
        result.push({
          role: 'tool',
          ...(part.callID ? { tool_call_id: part.callID } : {}),
          content: part.state.output || part.state.error || '',
        });
      }
    }
  }

  return result;
}

function buildTokenUsage(tokens: MessageInfo['tokens']): Record<string, number> | null {
  if (!tokens) {
    return null;
  }

  const inputTokens = tokens.input || 0;
  const outputTokens = tokens.output || 0;
  const reasoningTokens = tokens.reasoning || 0;

  const usage: Record<string, number> = {
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    total_tokens: inputTokens + outputTokens + reasoningTokens,
  };

  const cache = tokens.cache;
  if (cache) {
    if (cache.read) {
      usage.cache_read_tokens = cache.read;
    }
    if (cache.write) {
      usage.cache_write_tokens = cache.write;
    }
  }

  return usage;
}

function createLlmAndToolSpans(
  parentSpan: ReturnType<typeof startSpan>,
  messages: Message[],
  startIdx: number,
): void {
  for (let i = startIdx; i < messages.length; i++) {
    const msg = messages[i];
    if (msg.info.role !== MESSAGE_ROLE_ASSISTANT) {
      continue;
    }

    const parts = msg.parts;
    const modelId = msg.info.modelID || 'unknown';
    const providerId = msg.info.providerID || 'unknown';
    const tokens = msg.info.tokens;
    const timeInfo = msg.info.time || {};
    const createdNs = timestampToNs(timeInfo.created);
    const completedNs = timestampToNs(timeInfo.completed);

    const textParts = parts.filter((part) => part.type === PART_TYPE_TEXT);
    const toolParts = parts.filter((part) => part.type === PART_TYPE_TOOL);
    const reasoningParts = parts.filter((part) => part.type === PART_TYPE_REASONING);

    // Tool-call-only and reasoning-only responses are still LLM calls.
    if (textParts.length > 0 || toolParts.length > 0 || reasoningParts.length > 0) {
      const conversationMessages = reconstructConversationMessages(messages, i);
      const textContent = textParts.map((part) => part.text || '').join('\n');
      const reasoningText =
        reasoningParts.length > 0
          ? reasoningParts.map((part) => part.text || '').join('\n\n')
          : null;

      const llmSpan = startSpan({
        name: 'llm_call',
        parent: parentSpan,
        spanType: SpanType.LLM,
        startTimeNs: createdNs,
        inputs: {
          model: `${providerId}/${modelId}`,
          messages: conversationMessages,
        },
        attributes: {
          model: modelId,
          provider: providerId,
        },
      });

      const tokenUsage = buildTokenUsage(tokens);
      if (tokenUsage) {
        llmSpan.setAttribute(SpanAttributeKey.TOKEN_USAGE, tokenUsage);
      }

      const outputMessage: {
        role: string;
        content: string | null;
        reasoning?: string;
        tool_calls?: ToolCall[];
      } = { role: MESSAGE_ROLE_ASSISTANT, content: textContent || null };
      if (reasoningText) {
        outputMessage.reasoning = reasoningText;
      }
      if (toolParts.length > 0) {
        outputMessage.tool_calls = toolParts.map(toolCallFromPart);
      }

      llmSpan.setOutputs({ choices: [{ message: outputMessage }] });
      llmSpan.end({ endTimeNs: completedNs });
    }

    for (const toolPart of toolParts) {
      const state = toolPart.state || {};
      const toolName = toolPart.tool || 'unknown';
      const callId = toolPart.callID || '';
      const toolTime = state.time || {};
      const toolStartNs = timestampToNs(toolTime.start);
      const toolEndNs = timestampToNs(toolTime.end);

      const toolSpan = startSpan({
        name: `tool_${toolName}`,
        parent: parentSpan,
        spanType: SpanType.TOOL,
        startTimeNs: toolStartNs,
        inputs: state.input || {},
        attributes: {
          tool_name: toolName,
          tool_id: callId,
          status: state.status || 'unknown',
        },
      });

      if (state.status === 'completed') {
        toolSpan.setOutputs({
          result: state.output || '',
          title: state.title || '',
        });
      } else if (state.status === 'error') {
        toolSpan.setOutputs({
          error: state.error || 'Unknown error',
        });
      }

      toolSpan.end({ endTimeNs: toolEndNs });
    }
  }
}

function getTurnKey(messages: Message[], lastUserIdx: number): string {
  const userMessage = messages[lastUserIdx];
  if (userMessage.id) {
    return `message:${userMessage.id}`;
  }

  // V1 messages did not always carry IDs. Include the index and prompt so a
  // later appended turn is not mistaken for the earlier one.
  return `legacy:${lastUserIdx}:${extractUserPrompt([userMessage])}`;
}

function normalizeOutcomeStatus(value: unknown): SessionOutcomeStatus | undefined {
  const raw = typeof value === 'string' ? value : asString(asObject(value)?.type);
  switch (raw?.toLowerCase()) {
    case 'succeeded':
    case 'success':
    case 'completed':
      return 'succeeded';
    case 'failed':
    case 'error':
      return 'failed';
    case 'interrupted':
    case 'cancelled':
    case 'canceled':
      return 'interrupted';
    default:
      return undefined;
  }
}

function outcomeFromEvent(type: string, value: unknown): SessionOutcome | undefined {
  const data = asObject(value);
  if (type === 'session.execution.succeeded') {
    return { status: 'succeeded' };
  }
  if (type === 'session.execution.failed') {
    return { status: 'failed', error: errorToText(data?.error) };
  }
  if (type === 'session.execution.interrupted') {
    return { status: 'interrupted', reason: asString(data?.reason) };
  }
  if (type !== 'session.idle') {
    return undefined;
  }

  const status = normalizeOutcomeStatus(data?.outcome ?? data?.status);
  if (!status) {
    return undefined;
  }
  return {
    status,
    reason: asString(data?.reason),
    error: errorToText(data?.error),
  };
}

function getSessionID(event: { data?: unknown }): string | undefined {
  return asString(asObject(event.data)?.sessionID);
}

function getTraceStatus(
  messages: Message[],
  outcome?: SessionOutcome,
): {
  status: 'completed' | 'failed' | 'interrupted';
  error?: string;
  finish?: string;
  reason?: string;
} {
  const lastAssistant = [...messages]
    .reverse()
    .find((message) => message.info.role === MESSAGE_ROLE_ASSISTANT);
  const error = lastAssistant?.info.error;
  const finish = lastAssistant?.info.finish;

  if (outcome?.status === 'interrupted') {
    return {
      status: 'interrupted',
      ...(outcome.error || error ? { error: outcome.error || error } : {}),
      ...(finish ? { finish } : {}),
      ...(outcome.reason ? { reason: outcome.reason } : {}),
    };
  }
  if (outcome?.status === 'failed' || error || finish === 'error') {
    return {
      status: 'failed',
      ...(outcome?.error || error ? { error: outcome?.error || error } : {}),
      ...(finish ? { finish } : {}),
      ...(outcome?.reason ? { reason: outcome.reason } : {}),
    };
  }
  return {
    status: 'completed',
    ...(finish ? { finish } : {}),
    ...(outcome?.reason ? { reason: outcome.reason } : {}),
  };
}

/**
 * Process the most recent user turn. The normalized context is merged with
 * the session's retained ID-keyed transcript (because V2 context may be
 * post-compaction), while spans are emitted only for the latest turn.
 */
async function processSession(
  sessionId: string,
  messages: Message[],
  lastUserIdx: number,
  outcome?: SessionOutcome,
): Promise<boolean> {
  if (messages.length === 0) {
    if (DEBUG) {
      console.error('[mlflow] Empty messages list, skipping');
    }
    return false;
  }

  const turnMessages = messages.slice(lastUserIdx);
  const userPrompt = extractUserPrompt(turnMessages);
  if (!userPrompt) {
    if (DEBUG) {
      console.error('[mlflow] No user prompt text found');
    }
    return false;
  }

  if (DEBUG) {
    console.error(`[mlflow] Creating trace for session: ${sessionId}`);
  }

  const firstMsgTime = messages[lastUserIdx].info.time || {};
  const createdNs = timestampToNs(firstMsgTime.created);
  const lastMsg = messages[messages.length - 1];
  const lastMsgTime = lastMsg.info.time || {};
  const updatedNs = timestampToNs(lastMsgTime.completed || lastMsgTime.created);
  const finalResponse = extractAssistantResponse(turnMessages);
  const traceStatus = getTraceStatus(turnMessages, outcome);
  const fallbackResponse =
    traceStatus.error ||
    (traceStatus.status === 'interrupted'
      ? 'Conversation interrupted'
      : traceStatus.status === 'failed'
        ? 'Conversation failed'
        : 'Conversation completed');

  const traceId = await withSpan(
    (parentSpan) => {
      createLlmAndToolSpans(parentSpan, messages, lastUserIdx + 1);

      updateCurrentTrace({
        metadata: {
          [TRACE_SESSION_METADATA_KEY]: sessionId,
          [TRACE_USER_METADATA_KEY]: getCurrentUser(),
        },
        requestPreview: userPrompt.slice(0, MAX_PREVIEW_LENGTH),
        ...(finalResponse ? { responsePreview: finalResponse.slice(0, MAX_PREVIEW_LENGTH) } : {}),
      });

      parentSpan.setOutputs({
        response: finalResponse || fallbackResponse,
        status: traceStatus.status,
        ...(traceStatus.error ? { error: traceStatus.error } : {}),
        ...(traceStatus.finish ? { finish: traceStatus.finish } : {}),
        ...(traceStatus.reason ? { reason: traceStatus.reason } : {}),
      });
      parentSpan.end({ endTimeNs: updatedNs });

      return parentSpan.traceId;
    },
    {
      name: 'opencode_conversation',
      inputs: { prompt: userPrompt },
      startTimeNs: createdNs,
      spanType: SpanType.AGENT,
    },
  );

  await flushTraces();

  if (DEBUG) {
    console.error(`[mlflow] Created trace: ${traceId}`);
  }
  return true;
}

function touchSessionState(
  sessionStates: Map<string, SessionState>,
  sessionId: string,
  state: SessionState,
): void {
  sessionStates.delete(sessionId);
  sessionStates.set(sessionId, state);
}

function waitForDelay(signal: AbortSignal, delayMs: number): Promise<void> {
  return new Promise((resolve) => {
    if (signal.aborted) {
      resolve();
      return;
    }

    const finish = () => {
      clearTimeout(timer);
      signal.removeEventListener('abort', finish);
      resolve();
    };

    const timer = setTimeout(finish, delayMs);
    signal.addEventListener('abort', finish, { once: true });
  });
}

async function handleSessionIdle(
  ctx: Pick<PluginContext, 'session'>,
  sessionId: string,
  sessionStates: Map<string, SessionState>,
  signal: AbortSignal,
): Promise<void> {
  if (!ensureInitialized()) {
    return;
  }

  try {
    const rawMessages: unknown = await ctx.session.context({ sessionID: sessionId }, { signal });
    if (signal.aborted) {
      return;
    }

    const contextMessages = normalizeMessages(rawMessages);
    if (contextMessages.length === 0) {
      if (DEBUG) {
        console.error('[mlflow] Empty messages list, skipping');
      }
      return;
    }

    const state = sessionStates.get(sessionId) || {
      transcript: [],
      processedTurnKey: undefined,
      pending: false,
      queued: false,
    };
    const allMessages = mergeTranscript(state.transcript, contextMessages);
    state.transcript = allMessages;
    touchSessionState(sessionStates, sessionId, state);

    if (allMessages.length === 0) {
      if (DEBUG) {
        console.error('[mlflow] Empty messages list, skipping');
      }
      return;
    }

    const lastUserIdx = findLastUserMessageIndex(allMessages);
    if (lastUserIdx == null) {
      if (DEBUG) {
        console.error('[mlflow] No user message found in session');
      }
      return;
    }

    const turnKey = getTurnKey(allMessages, lastUserIdx);
    if (state.processedTurnKey === turnKey) {
      return;
    }

    if (signal.aborted) {
      return;
    }

    const traceOutcome = state.outcome;
    if (await processSession(sessionId, allMessages, lastUserIdx, traceOutcome)) {
      if (signal.aborted) {
        return;
      }
      state.processedTurnKey = turnKey;
      if (state.outcome === traceOutcome) {
        state.outcome = undefined;
      }
    }
  } catch (error) {
    if (DEBUG && !signal.aborted) {
      console.error('[mlflow] Error processing session:', error);
    }
  }
}

/**
 * MLflow tracing plugin for OpenCode V2.
 * Automatically traces conversations to MLflow when sessions complete.
 */
export const MLflowTracingPlugin = Plugin.define({
  id: 'mlflow-opencode',
  setup(ctx) {
    const controller = new AbortController();
    const sessionStates = new Map<string, SessionState>();
    const inFlight = new Map<string, Promise<void>>();
    const pendingSessions = new Set<string>();
    let activeSessions = 0;
    let stopped = false;

    const getState = (sessionId: string): SessionState => {
      const existing = sessionStates.get(sessionId);
      if (existing) {
        return existing;
      }

      const state: SessionState = {
        transcript: [],
        processedTurnKey: undefined,
        pending: false,
        queued: false,
      };
      sessionStates.set(sessionId, state);
      return state;
    };

    const drain = (): void => {
      while (!stopped && !controller.signal.aborted && activeSessions < MAX_CONCURRENT_SESSIONS) {
        const sessionId = pendingSessions.values().next().value;
        if (sessionId === undefined) {
          return;
        }

        pendingSessions.delete(sessionId);
        const state = getState(sessionId);
        state.queued = false;
        if (!state.pending) {
          touchSessionState(sessionStates, sessionId, state);
          continue;
        }
        state.pending = false;
        activeSessions += 1;
        const work = handleSessionIdle(ctx, sessionId, sessionStates, controller.signal);
        state.inFlight = work;
        inFlight.set(sessionId, work);

        const finish = (): void => {
          activeSessions -= 1;
          if (state.inFlight === work) {
            state.inFlight = undefined;
          }
          if (inFlight.get(sessionId) === work) {
            inFlight.delete(sessionId);
          }
          if (!stopped && !controller.signal.aborted && state.pending && !state.queued) {
            state.queued = true;
            pendingSessions.add(sessionId);
          }
          if (!stopped) {
            touchSessionState(sessionStates, sessionId, state);
            drain();
          }
        };

        // The event stream must remain responsive while context retrieval,
        // span export, or network retries are in progress.
        void work.then(finish, (error) => {
          if (DEBUG && !controller.signal.aborted) {
            console.error('[mlflow] Error processing session:', error);
          }
          finish();
        });
      }
    };

    const enqueue = (sessionId: string): void => {
      if (stopped || controller.signal.aborted) {
        return;
      }

      const state = getState(sessionId);
      state.pending = true;
      if (!state.queued && !state.inFlight) {
        state.queued = true;
        pendingSessions.add(sessionId);
      }
      touchSessionState(sessionStates, sessionId, state);
      drain();
    };

    const recordEvent = (event: { type: string; data?: unknown }): void => {
      const sessionId = getSessionID(event);
      if (!sessionId) {
        return;
      }

      const state = getState(sessionId);
      if (event.type === 'session.execution.started') {
        state.outcome = undefined;
      } else {
        const outcome = outcomeFromEvent(event.type, event.data);
        if (outcome) {
          state.outcome = outcome;
        }
      }
      touchSessionState(sessionStates, sessionId, state);

      if (
        event.type === 'session.idle' ||
        event.type === 'session.execution.succeeded' ||
        event.type === 'session.execution.failed' ||
        event.type === 'session.execution.interrupted'
      ) {
        enqueue(sessionId);
      }
    };

    const subscribe = async (): Promise<void> => {
      let retryDelay = SUBSCRIPTION_RETRY_INITIAL_MS;
      while (!controller.signal.aborted) {
        try {
          for await (const event of ctx.event.subscribe({ signal: controller.signal })) {
            retryDelay = SUBSCRIPTION_RETRY_INITIAL_MS;
            recordEvent(event);
          }
        } catch (error) {
          if (controller.signal.aborted) {
            break;
          }
          if (DEBUG) {
            console.error('[mlflow] OpenCode event subscription ended:', error);
          }
        }

        if (controller.signal.aborted) {
          break;
        }
        await waitForDelay(controller.signal, retryDelay);
        retryDelay = Math.min(retryDelay * 2, SUBSCRIPTION_RETRY_MAX_MS);
      }
    };

    void subscribe();

    return async () => {
      stopped = true;
      controller.abort();
      pendingSessions.clear();
      for (const state of sessionStates.values()) {
        state.pending = false;
        state.queued = false;
      }
      const inFlightWork = Promise.allSettled(Array.from(inFlight.values()));
      await new Promise<void>((resolve) => {
        const cleanupTimer = setTimeout(resolve, CLEANUP_TIMEOUT_MS);
        void inFlightWork.then(() => {
          clearTimeout(cleanupTimer);
          resolve();
        });
      });
      inFlight.clear();
      sessionStates.clear();
    };
  },
});

export default MLflowTracingPlugin;
