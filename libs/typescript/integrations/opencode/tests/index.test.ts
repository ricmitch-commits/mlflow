/** Tests for the OpenCode V2 MLflow integration. */

import type { PluginContext, PluginEvent } from '@opencode/plugin';
import MLflowTracingPlugin from '../src';

jest.mock('@mlflow/core', () => {
  const createSpan = () => ({
    traceId: 'mock-trace-id',
    setAttribute: jest.fn(),
    setOutputs: jest.fn(),
    end: jest.fn(),
  });
  const parentSpan = createSpan();

  return {
    getParentSpan: jest.fn(() => parentSpan),
    init: jest.fn(),
    startSpan: jest.fn(createSpan),
    withSpan: jest.fn((callback: (span: ReturnType<typeof createSpan>) => unknown) =>
      callback(parentSpan),
    ),
    updateCurrentTrace: jest.fn(),
    flushTraces: jest.fn().mockResolvedValue(undefined),
    SpanType: {
      LLM: 'LLM',
      TOOL: 'TOOL',
      AGENT: 'AGENT',
    },
    SpanAttributeKey: {
      TOKEN_USAGE: 'token_usage',
    },
  };
});

import * as mlflowTracing from '@mlflow/core';

type MockSpan = {
  setOutputs: jest.Mock;
  setAttribute: jest.Mock;
  end: jest.Mock;
};

function getParentSpan(): MockSpan {
  return (mlflowTracing as unknown as { getParentSpan: jest.Mock }).getParentSpan();
}

interface EventBus {
  iterable: AsyncIterable<PluginEvent>;
  emit: (event: PluginEvent) => void;
  close: () => void;
}

function createEventBus(): EventBus {
  const queue: PluginEvent[] = [];
  const waiters: Array<(event: IteratorResult<PluginEvent>) => void> = [];
  let closed = false;

  const next = (): Promise<IteratorResult<PluginEvent>> => {
    if (queue.length > 0) {
      return Promise.resolve({ done: false, value: queue.shift() as PluginEvent });
    }
    if (closed) {
      return Promise.resolve({ done: true, value: undefined });
    }
    return new Promise((resolve) => waiters.push(resolve));
  };

  return {
    iterable: {
      [Symbol.asyncIterator]: () => ({ next }),
    },
    emit(event) {
      const waiter = waiters.shift();
      if (waiter) {
        waiter({ done: false, value: event });
      } else {
        queue.push(event);
      }
    },
    close() {
      closed = true;
      for (const waiter of waiters.splice(0)) {
        waiter({ done: true, value: undefined });
      }
    },
  };
}

function createUserMessage(text: string, id: string, created = 1) {
  return {
    id,
    type: 'user',
    text,
    time: { created },
  };
}

function createAssistantMessage(
  id: string,
  text: string,
  options: {
    modelID?: string;
    providerID?: string;
    tokens?: Record<string, unknown>;
    created?: number;
    completed?: number;
    finish?: string;
    error?: unknown;
  } = {},
) {
  return {
    id,
    type: 'assistant',
    model: {
      id: options.modelID || 'claude-3-opus',
      providerID: options.providerID || 'anthropic',
    },
    tokens: options.tokens || {
      input: 100,
      output: 50,
      reasoning: 0,
      cache: { read: 0, write: 0 },
    },
    time: {
      created: options.created || 2,
      completed: options.completed || 3,
    },
    ...(options.finish ? { finish: options.finish } : {}),
    ...(options.error ? { error: options.error } : {}),
    content: [{ type: 'text', text }],
  };
}

function createToolAssistantMessage() {
  return {
    id: 'assistant-tool-1',
    type: 'assistant',
    model: { id: 'gpt-4.1', providerID: 'openai' },
    tokens: {
      input: 150,
      output: 40,
      reasoning: 10,
      cache: { read: 25, write: 5 },
    },
    time: { created: 10, completed: 20 },
    content: [
      { type: 'reasoning', text: 'I should inspect the file first.' },
      {
        type: 'tool',
        id: 'call-read-1',
        name: 'Read',
        executed: true,
        time: { created: 11, ran: 12, completed: 18 },
        state: {
          status: 'completed',
          input: { file_path: '/tmp/example.ts' },
          content: [{ type: 'text', text: 'const answer = 42;' }],
        },
      },
    ],
  };
}

function createContext(transcript: unknown): {
  context: PluginContext;
  bus: EventBus;
  sessionContext: jest.Mock;
  subscribe: jest.Mock;
} {
  const bus = createEventBus();
  const sessionContext = jest.fn().mockResolvedValue(transcript);
  const subscribe = jest.fn(({ signal }: { signal: AbortSignal }) => {
    signal.addEventListener('abort', bus.close, { once: true });
    return bus.iterable;
  });

  return {
    context: {
      event: { subscribe },
      session: { context: sessionContext },
    },
    bus,
    sessionContext,
    subscribe,
  };
}

async function waitFor(predicate: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 20; attempt += 1) {
    if (predicate()) {
      return;
    }
    await new Promise<void>((resolve) => setImmediate(resolve));
  }
  throw new Error('Timed out waiting for the OpenCode event subscription');
}

async function emitIdle(bus: EventBus, sessionID: string, expectedCalls: number): Promise<void> {
  bus.emit({ type: 'session.idle', data: { sessionID } });
  await waitFor(() => (mlflowTracing.withSpan as jest.Mock).mock.calls.length >= expectedCalls);
}

describe('MLflow OpenCode V2 plugin', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    jest.clearAllMocks();
    process.env = { ...originalEnv };
    delete process.env.MLFLOW_TRACKING_URI;
    delete process.env.MLFLOW_EXPERIMENT_ID;
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it('exports a stable V2 plugin definition and cleans up its subscription', async () => {
    const { context, subscribe } = createContext([]);

    expect(MLflowTracingPlugin.id).toBe('mlflow-opencode');
    const cleanup = MLflowTracingPlugin.setup(context);

    expect(subscribe).toHaveBeenCalledWith({ signal: expect.any(AbortSignal) });
    const signal = subscribe.mock.calls[0][0].signal as AbortSignal;
    expect(signal.aborted).toBe(false);

    await (cleanup as () => void | Promise<void>)();
    expect(signal.aborted).toBe(true);
  });

  it('uses V2 event data and session context, ignoring unrelated events', async () => {
    process.env.MLFLOW_TRACKING_URI = 'http://localhost:5000';
    process.env.MLFLOW_EXPERIMENT_ID = '123';
    const { context, bus, sessionContext } = createContext([
      createUserMessage('Hello', 'user-1'),
      createAssistantMessage('assistant-1', 'Hi there!'),
    ]);
    MLflowTracingPlugin.setup(context);

    bus.emit({ type: 'message.updated', data: { sessionID: 'ignored' } });
    bus.emit({ type: 'session.idle', data: {} });
    await new Promise<void>((resolve) => setImmediate(resolve));
    expect(sessionContext).not.toHaveBeenCalled();

    await emitIdle(bus, 'session-1', 1);
    expect(sessionContext).toHaveBeenCalledWith(
      { sessionID: 'session-1' },
      { signal: expect.any(AbortSignal) },
    );
    expect(mlflowTracing.withSpan).toHaveBeenCalledWith(
      expect.any(Function),
      expect.objectContaining({
        name: 'opencode_conversation',
        spanType: 'AGENT',
        inputs: { prompt: 'Hello' },
      }),
    );
  });

  it('traces a successful execution when OpenCode omits session.idle', async () => {
    process.env.MLFLOW_TRACKING_URI = 'http://localhost:5000';
    process.env.MLFLOW_EXPERIMENT_ID = '123';
    const { context, bus, sessionContext } = createContext([
      createUserMessage('Complete this turn', 'user-complete'),
      createAssistantMessage('assistant-complete', 'Done'),
    ]);
    MLflowTracingPlugin.setup(context);

    bus.emit({
      type: 'session.execution.succeeded',
      data: { sessionID: 'completed-session' },
    });
    await waitFor(() => sessionContext.mock.calls.length === 1);
    expect(mlflowTracing.withSpan).toHaveBeenCalledTimes(1);
  });

  it('traces V2 assistant model, token, reasoning, and tool shapes', async () => {
    process.env.MLFLOW_TRACKING_URI = 'http://localhost:5000';
    process.env.MLFLOW_EXPERIMENT_ID = '123';
    const { context, bus } = createContext([
      createUserMessage('Read the file', 'user-tool'),
      createToolAssistantMessage(),
    ]);
    MLflowTracingPlugin.setup(context);

    await emitIdle(bus, 'tool-session', 1);

    expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
      expect.objectContaining({
        name: 'llm_call',
        spanType: 'LLM',
        inputs: expect.objectContaining({ model: 'openai/gpt-4.1' }),
      }),
    );
    expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
      expect.objectContaining({
        name: 'tool_Read',
        spanType: 'TOOL',
        startTimeNs: 12 * 1e6,
        inputs: { file_path: '/tmp/example.ts' },
        attributes: expect.objectContaining({ status: 'completed', tool_id: 'call-read-1' }),
      }),
    );

    const llmCall = (mlflowTracing.startSpan as jest.Mock).mock.calls.find(
      ([options]) => options.name === 'llm_call',
    );
    const llmSpan = (mlflowTracing.startSpan as jest.Mock).mock.results.find(
      (_result, index) =>
        (mlflowTracing.startSpan as jest.Mock).mock.calls[index][0].name === 'llm_call',
    )?.value;
    expect(llmCall?.[0].inputs.messages).toEqual([{ role: 'user', content: 'Read the file' }]);
    expect(llmSpan.setAttribute).toHaveBeenCalledWith('token_usage', {
      input_tokens: 150,
      output_tokens: 40,
      total_tokens: 200,
      cache_read_tokens: 25,
      cache_write_tokens: 5,
    });
    expect(llmSpan.setOutputs).toHaveBeenCalledWith({
      choices: [
        {
          message: expect.objectContaining({
            content: null,
            reasoning: 'I should inspect the file first.',
            tool_calls: [
              expect.objectContaining({
                id: 'call-read-1',
                function: expect.objectContaining({ name: 'Read' }),
              }),
            ],
          }),
        },
      ],
    });
  });

  it('preserves V2 structured tool errors and completion timing', async () => {
    process.env.MLFLOW_TRACKING_URI = 'http://localhost:5000';
    process.env.MLFLOW_EXPERIMENT_ID = '123';
    const { context, bus } = createContext([
      createUserMessage('Run the command', 'user-error'),
      {
        id: 'assistant-error-1',
        type: 'assistant',
        model: { id: 'claude-3-opus', providerID: 'anthropic' },
        time: { created: 20, completed: 30 },
        content: [
          {
            type: 'tool',
            id: 'call-bash-1',
            name: 'Bash',
            time: { created: 21, ran: 22, completed: 29 },
            state: {
              status: 'error',
              input: { command: 'not-a-command' },
              error: { type: 'ProcessError', message: 'Command not found' },
              content: [{ type: 'text', text: 'The command failed.' }],
            },
          },
        ],
      },
    ]);
    MLflowTracingPlugin.setup(context);

    await emitIdle(bus, 'error-session', 1);

    expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
      expect.objectContaining({
        name: 'tool_Bash',
        startTimeNs: 22 * 1e6,
        attributes: expect.objectContaining({ status: 'error' }),
      }),
    );
    const toolSpan = (mlflowTracing.startSpan as jest.Mock).mock.results.find(
      (_result, index) =>
        (mlflowTracing.startSpan as jest.Mock).mock.calls[index][0].name === 'tool_Bash',
    )?.value;
    expect(toolSpan.setOutputs).toHaveBeenCalledWith({ error: 'Command not found' });
    expect(toolSpan.end).toHaveBeenCalledWith({ endTimeNs: 29 * 1e6 });
  });

  it('preserves assistant finish and error on failed parent traces', async () => {
    process.env.MLFLOW_TRACKING_URI = 'http://localhost:5000';
    process.env.MLFLOW_EXPERIMENT_ID = '123';
    const { context, bus } = createContext([
      createUserMessage('Try the request', 'user-failed'),
      createAssistantMessage('assistant-failed', 'The request stopped.', {
        finish: 'error',
        error: { type: 'ProviderError', message: 'Provider unavailable' },
      }),
    ]);
    MLflowTracingPlugin.setup(context);

    bus.emit({
      type: 'session.execution.failed',
      data: { sessionID: 'failed-session', error: { message: 'Execution failed' } },
    });
    await emitIdle(bus, 'failed-session', 1);

    expect(getParentSpan().setOutputs).toHaveBeenCalledWith({
      response: 'The request stopped.',
      status: 'failed',
      error: 'Execution failed',
      finish: 'error',
    });
  });

  it('preserves an interrupted session.idle outcome on the parent trace', async () => {
    process.env.MLFLOW_TRACKING_URI = 'http://localhost:5000';
    process.env.MLFLOW_EXPERIMENT_ID = '123';
    const { context, bus } = createContext([
      createUserMessage('Stop this work', 'user-interrupted'),
      createAssistantMessage('assistant-interrupted', 'Partial response', { finish: 'stop' }),
    ]);
    MLflowTracingPlugin.setup(context);

    bus.emit({
      type: 'session.idle',
      data: { sessionID: 'interrupted-session', outcome: 'interrupted', reason: 'user' },
    });
    await emitIdle(bus, 'interrupted-session', 1);

    expect(getParentSpan().setOutputs).toHaveBeenCalledWith({
      response: 'Partial response',
      status: 'interrupted',
      finish: 'stop',
      reason: 'user',
    });
  });

  it('preserves prior V2 messages in incremental conversation history', async () => {
    process.env.MLFLOW_TRACKING_URI = 'http://localhost:5000';
    process.env.MLFLOW_EXPERIMENT_ID = '123';
    let transcript: unknown[] = [
      createUserMessage('First question', 'user-1'),
      createAssistantMessage('assistant-1', 'First answer'),
    ];
    const { context, bus, sessionContext } = createContext(transcript);
    sessionContext.mockImplementation(() => Promise.resolve(transcript));
    MLflowTracingPlugin.setup(context);

    await emitIdle(bus, 'incremental-session', 1);
    transcript = [
      ...transcript,
      createUserMessage('Second question', 'user-2', 4),
      createAssistantMessage('assistant-2', 'Second answer', { created: 5, completed: 6 }),
    ];
    await emitIdle(bus, 'incremental-session', 2);

    const llmCalls = (mlflowTracing.startSpan as jest.Mock).mock.calls.filter(
      ([options]) => options.name === 'llm_call',
    );
    expect(llmCalls).toHaveLength(2);
    expect(llmCalls[1][0].inputs.messages).toEqual([
      { role: 'user', content: 'First question' },
      { role: 'assistant', content: 'First answer' },
      { role: 'user', content: 'Second question' },
    ]);
  });

  it('merges post-compaction V2 context with the retained ID-keyed transcript', async () => {
    process.env.MLFLOW_TRACKING_URI = 'http://localhost:5000';
    process.env.MLFLOW_EXPERIMENT_ID = '123';
    let contextMessages: unknown[] = [
      createUserMessage('Before compaction', 'user-before'),
      createAssistantMessage('assistant-before', 'Earlier answer'),
    ];
    const { context, bus, sessionContext } = createContext(contextMessages);
    sessionContext.mockImplementation(() => Promise.resolve(contextMessages));
    MLflowTracingPlugin.setup(context);

    await emitIdle(bus, 'compaction-session', 1);
    contextMessages = [
      createUserMessage('After compaction', 'user-after', 4),
      createAssistantMessage('assistant-after', 'Later answer', { created: 5, completed: 6 }),
    ];
    await emitIdle(bus, 'compaction-session', 2);

    const llmCalls = (mlflowTracing.startSpan as jest.Mock).mock.calls.filter(
      ([options]) => options.name === 'llm_call',
    );
    expect(llmCalls[1][0].inputs.messages).toEqual([
      { role: 'user', content: 'Before compaction' },
      { role: 'assistant', content: 'Earlier answer' },
      { role: 'user', content: 'After compaction' },
    ]);
  });

  it('creates a meaningful prompt for a file-only user message', async () => {
    process.env.MLFLOW_TRACKING_URI = 'http://localhost:5000';
    process.env.MLFLOW_EXPERIMENT_ID = '123';
    const { context, bus } = createContext([
      {
        id: 'user-file-only',
        type: 'user',
        text: '',
        files: [{ name: 'report.csv', mime: 'text/csv', description: 'the uploaded report' }],
        time: { created: 1 },
      },
      createAssistantMessage('assistant-file-only', 'I will inspect the report.'),
    ]);
    MLflowTracingPlugin.setup(context);

    await emitIdle(bus, 'file-only-session', 1);

    expect(mlflowTracing.withSpan).toHaveBeenCalledWith(
      expect.any(Function),
      expect.objectContaining({
        inputs: { prompt: 'Attached files: report.csv: the uploaded report' },
      }),
    );
    expect(mlflowTracing.updateCurrentTrace).toHaveBeenCalledWith(
      expect.objectContaining({
        requestPreview: 'Attached files: report.csv: the uploaded report',
      }),
    );
  });

  it('keeps consuming events while MLflow export is slow', async () => {
    process.env.MLFLOW_TRACKING_URI = 'http://localhost:5000';
    process.env.MLFLOW_EXPERIMENT_ID = '123';
    const { context, bus, sessionContext } = createContext([
      createUserMessage('Slow export', 'user-slow'),
      createAssistantMessage('assistant-slow', 'Working'),
    ]);
    const releaseExports: Array<() => void> = [];
    (mlflowTracing.flushTraces as jest.Mock).mockImplementation(
      () =>
        new Promise<void>((resolve) => {
          releaseExports.push(resolve);
        }),
    );
    const cleanup = MLflowTracingPlugin.setup(context);

    bus.emit({ type: 'session.idle', data: { sessionID: 'slow-session-1' } });
    await waitFor(() => sessionContext.mock.calls.length === 1);
    bus.emit({ type: 'session.idle', data: { sessionID: 'slow-session-2' } });
    await waitFor(() => sessionContext.mock.calls.length === 2);

    expect(sessionContext).toHaveBeenCalledTimes(2);
    for (const releaseExport of releaseExports) {
      releaseExport();
    }
    await (cleanup as () => void | Promise<void>)();
  });

  it('reconnects the event subscription after it ends', async () => {
    process.env.MLFLOW_TRACKING_URI = 'http://localhost:5000';
    process.env.MLFLOW_EXPERIMENT_ID = '123';
    const { context, subscribe } = createContext([
      createUserMessage('Reconnect', 'user-reconnect'),
      createAssistantMessage('assistant-reconnect', 'Reconnected'),
    ]);
    const first = createEventBus();
    const second = createEventBus();
    let subscriptions = 0;
    subscribe.mockImplementation(({ signal }: { signal: AbortSignal }) => {
      const bus = subscriptions++ === 0 ? first : second;
      signal.addEventListener('abort', bus.close, { once: true });
      return bus.iterable;
    });
    const cleanup = MLflowTracingPlugin.setup(context);
    expect(subscribe).toHaveBeenCalledTimes(1);

    first.close();
    await new Promise<void>((resolve) => setTimeout(resolve, 150));
    expect(subscribe.mock.calls.length).toBeGreaterThanOrEqual(2);

    second.emit({ type: 'session.idle', data: { sessionID: 'reconnected-session' } });
    await waitFor(() => (mlflowTracing.withSpan as jest.Mock).mock.calls.length === 1);
    await (cleanup as () => void | Promise<void>)();
  });

  it('aborts context retrieval during bounded cleanup', async () => {
    process.env.MLFLOW_TRACKING_URI = 'http://localhost:5000';
    process.env.MLFLOW_EXPERIMENT_ID = '123';
    const { context, bus, sessionContext } = createContext([]);
    let requestSignal: AbortSignal | undefined;
    sessionContext.mockImplementation(
      (_input: unknown, options?: { signal?: AbortSignal }) =>
        new Promise<unknown[]>((_resolve, reject) => {
          requestSignal = options?.signal;
          options?.signal?.addEventListener('abort', () => reject(new Error('aborted')), {
            once: true,
          });
        }),
    );
    const cleanup = MLflowTracingPlugin.setup(context);
    bus.emit({ type: 'session.idle', data: { sessionID: 'cancel-session' } });
    await waitFor(() => requestSignal !== undefined);

    await (cleanup as () => void | Promise<void>)();
    expect(requestSignal?.aborted).toBe(true);
  });

  it('does not create duplicate traces for repeated idle events', async () => {
    process.env.MLFLOW_TRACKING_URI = 'http://localhost:5000';
    process.env.MLFLOW_EXPERIMENT_ID = '123';
    const { context, bus } = createContext([
      createUserMessage('Repeat-safe', 'stable-user-id'),
      createAssistantMessage('Only once', 'stable-assistant-id'),
    ]);
    MLflowTracingPlugin.setup(context);

    await emitIdle(bus, 'duplicate-session', 1);
    bus.emit({ type: 'session.idle', data: { sessionID: 'duplicate-session' } });
    await new Promise<void>((resolve) => setImmediate(resolve));
    await new Promise<void>((resolve) => setImmediate(resolve));

    expect(mlflowTracing.withSpan).toHaveBeenCalledTimes(1);
  });
});
