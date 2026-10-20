// Minimal V2 @opencode/plugin mock used by the focused integration tests.
export interface PluginEvent {
  type: string;
  data: Record<string, unknown>;
}

export interface PluginContext {
  event: {
    subscribe: (options: { signal: AbortSignal }) => AsyncIterable<PluginEvent>;
  };
  session: {
    context: (
      input: { sessionID: string },
      options?: { signal?: AbortSignal },
    ) => Promise<readonly unknown[]>;
  };
}

export interface PluginDefinition {
  id: string;
  setup: (context: PluginContext) => void | (() => void | Promise<void>);
}

export const Plugin = {
  define(definition: PluginDefinition): PluginDefinition {
    return definition;
  },
};
