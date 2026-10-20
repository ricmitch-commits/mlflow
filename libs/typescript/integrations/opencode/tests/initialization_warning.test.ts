import type { PluginContext, PluginEvent } from '@opencode/plugin';
import MLflowTracingPlugin from '../src';

jest.mock('@mlflow/core', () => ({
  init: jest.fn(),
  startSpan: jest.fn(),
  withSpan: jest.fn(),
  updateCurrentTrace: jest.fn(),
  flushTraces: jest.fn().mockResolvedValue(undefined),
  SpanType: { LLM: 'LLM', TOOL: 'TOOL', AGENT: 'AGENT' },
  SpanAttributeKey: { TOKEN_USAGE: 'token_usage' },
}));

import * as mlflowTracing from '@mlflow/core';

function createEventSource() {
  let resolveNext: ((result: IteratorResult<PluginEvent>) => void) | undefined;
  let closed = false;
  const iterable: AsyncIterable<PluginEvent> = {
    [Symbol.asyncIterator]() {
      return {
        next: () => {
          if (closed) {
            return Promise.resolve({ done: true, value: undefined });
          }
          return new Promise<IteratorResult<PluginEvent>>((resolve) => {
            resolveNext = resolve;
          });
        },
      };
    },
  };

  return {
    iterable,
    emit(event: PluginEvent) {
      resolveNext?.({ done: false, value: event });
      resolveNext = undefined;
    },
    close() {
      closed = true;
      resolveNext?.({ done: true, value: undefined });
      resolveNext = undefined;
    },
  };
}

function createContext(messages: unknown[]) {
  const source = createEventSource();
  const sessionContext = jest.fn().mockResolvedValue(messages);
  const context: PluginContext = {
    event: {
      subscribe: ({ signal }) => {
        signal.addEventListener('abort', source.close, { once: true });
        return source.iterable;
      },
    },
    session: { context: sessionContext },
  };
  return { context, source, sessionContext };
}

async function waitFor(predicate: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 20; attempt += 1) {
    if (predicate()) {
      return;
    }
    await new Promise<void>((resolve) => setImmediate(resolve));
  }
  throw new Error('Timed out waiting for OpenCode event processing');
}

describe('OpenCode initialization failure', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    jest.clearAllMocks();
    process.env = { ...originalEnv };
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it('warns once and retries initialization on later idle events', async () => {
    process.env.MLFLOW_TRACKING_URI = 'databricks';
    process.env.MLFLOW_EXPERIMENT_ID = '123';
    process.env.MLFLOW_TRACE_LOCATION = 'invalid';
    (mlflowTracing.init as jest.Mock).mockImplementation(() => {
      throw new Error('Invalid MLFLOW_TRACE_LOCATION: expected catalog.schema.table_prefix');
    });
    const warning = jest.spyOn(console, 'error').mockImplementation(() => {});
    const { context, source, sessionContext } = createContext([]);
    MLflowTracingPlugin.setup(context);

    try {
      source.emit({ type: 'session.idle', data: { sessionID: 'session-1' } });
      await waitFor(() => (mlflowTracing.init as jest.Mock).mock.calls.length === 1);
      source.emit({ type: 'session.idle', data: { sessionID: 'session-1' } });
      await waitFor(() => (mlflowTracing.init as jest.Mock).mock.calls.length === 2);

      expect(sessionContext).not.toHaveBeenCalled();
      expect(warning).toHaveBeenCalledTimes(1);
      expect(warning).toHaveBeenCalledWith(
        '[mlflow] OpenCode tracing is disabled: Invalid MLFLOW_TRACE_LOCATION: expected catalog.schema.table_prefix',
      );
    } finally {
      warning.mockRestore();
    }
  });

  it('resumes V2 context handling after a transient initialization failure', async () => {
    process.env.MLFLOW_TRACKING_URI = 'databricks';
    process.env.MLFLOW_EXPERIMENT_ID = '123';
    (mlflowTracing.init as jest.Mock)
      .mockImplementationOnce(() => {
        throw new Error('temporary initialization failure');
      })
      .mockImplementationOnce(() => undefined);
    const warning = jest.spyOn(console, 'error').mockImplementation(() => {});
    const { context, source, sessionContext } = createContext([]);
    MLflowTracingPlugin.setup(context);

    try {
      source.emit({ type: 'session.idle', data: { sessionID: 'session-2' } });
      await waitFor(() => (mlflowTracing.init as jest.Mock).mock.calls.length === 1);
      expect(sessionContext).not.toHaveBeenCalled();

      source.emit({ type: 'session.idle', data: { sessionID: 'session-2' } });
      await waitFor(() => sessionContext.mock.calls.length === 1);
      expect(mlflowTracing.init).toHaveBeenCalledTimes(2);
      expect(warning).toHaveBeenCalledTimes(1);
    } finally {
      warning.mockRestore();
    }
  });
});
