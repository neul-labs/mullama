// Mullama API Client

const BASE_URL = ''  // Same origin

export interface Model {
  id: string
  object: string
  created: number
  owned_by: string
}

export interface ModelsResponse {
  object: string
  data: Model[]
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant'
  content: string
}

export interface ChatRequest {
  model: string
  messages: ChatMessage[]
  stream?: boolean
  max_tokens?: number
  temperature?: number
}

export interface ChatChoice {
  index: number
  message: ChatMessage
  finish_reason: string
}

export interface ChatResponse {
  id: string
  object: string
  created: number
  model: string
  choices: ChatChoice[]
  usage?: {
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
  }
}

export interface SystemStatus {
  uptime_secs: number
  version: string
  models_loaded: number
  http_endpoint?: string
}

export interface ModelDetails {
  name: string
  filename: string
  path: string
  repo_id?: string
  size: number
  size_formatted: string
  loaded: boolean
  downloaded?: string
  source?: string
}

export interface PullProgress {
  status: string
  progress?: number
  total?: number
  speed?: string
}

export interface DefaultModel {
  name: string
  description: string
  size_hint: string
  tags: string[]
  from: string
  has_thinking: boolean
  has_vision: boolean
  has_tools: boolean
}

export interface DefaultsResponse {
  models: DefaultModel[]
}

// Fetch wrapper with error handling
async function fetchApi<T>(
  path: string,
  options?: RequestInit
): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: { message: response.statusText } }))
    throw new Error(error.error?.message || `HTTP ${response.status}`)
  }

  return response.json()
}

// OpenAI-compatible endpoints
export const openai = {
  async listModels(): Promise<Model[]> {
    const res = await fetchApi<ModelsResponse>('/v1/models')
    return res.data
  },

  async chat(request: ChatRequest): Promise<ChatResponse> {
    return fetchApi<ChatResponse>('/v1/chat/completions', {
      method: 'POST',
      body: JSON.stringify({ ...request, stream: false }),
    })
  },

  // Streaming chat that calls onChunk for each token and returns when done
  async chatStream(
    request: ChatRequest,
    onChunk: (content: string, thinking?: string) => void
  ): Promise<void> {
    const response = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...request, stream: true }),
    })

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }

    const reader = response.body?.getReader()
    if (!reader) throw new Error('No response body')

    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()

      if (done) {
        // Process any remaining data in buffer
        if (buffer.trim()) {
          processLines(buffer, onChunk)
        }
        break
      }

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6).trim()
          if (data === '[DONE]') {
            return
          }
          if (data) {
            try {
              const parsed = JSON.parse(data)
              const content = parsed.choices?.[0]?.delta?.content
              const thinking = parsed.choices?.[0]?.delta?.thinking
              if (content || thinking) {
                onChunk(content || '', thinking)
              }
            } catch {
              // Ignore parse errors
            }
          }
        }
      }
    }
  },
}

function processLines(buffer: string, onChunk: (content: string, thinking?: string) => void) {
  const lines = buffer.split('\n')
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = line.slice(6).trim()
      if (data && data !== '[DONE]') {
        try {
          const parsed = JSON.parse(data)
          const content = parsed.choices?.[0]?.delta?.content
          const thinking = parsed.choices?.[0]?.delta?.thinking
          if (content || thinking) {
            onChunk(content || '', thinking)
          }
        } catch {
          // Ignore parse errors
        }
      }
    }
  }
}

// Management API endpoints
export const management = {
  async status(): Promise<SystemStatus> {
    return fetchApi<SystemStatus>('/api/system/status')
  },

  async listModels(): Promise<ModelDetails[]> {
    const response = await fetchApi<{ models: ModelDetails[], available_aliases: string[], total_cached: number }>('/api/models')
    return response.models
  },

  async getModel(name: string): Promise<ModelDetails> {
    return fetchApi<ModelDetails>(`/api/models/${encodeURIComponent(name)}`)
  },

  async pullModel(
    name: string,
    onProgress?: (progress: PullProgress) => void
  ): Promise<void> {
    const response = await fetch('/api/models/pull', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: name }),
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: { message: response.statusText } }))
      throw new Error(error.error?.message || `HTTP ${response.status}`)
    }

    // Handle streaming progress if supported
    const reader = response.body?.getReader()
    if (reader && onProgress) {
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.trim()) {
            try {
              const progress = JSON.parse(line) as PullProgress
              onProgress(progress)
            } catch {
              // Ignore parse errors
            }
          }
        }
      }
    }
  },

  async deleteModel(name: string): Promise<void> {
    await fetchApi<void>(`/api/models/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    })
  },

  async loadModel(name: string, options?: { gpu_layers?: number; context_size?: number }): Promise<{ success: boolean; message: string; model?: any }> {
    return fetchApi<{ success: boolean; message: string; model?: any }>('/api/models/load', {
      method: 'POST',
      body: JSON.stringify({
        name,
        gpu_layers: options?.gpu_layers,
        context_size: options?.context_size,
      }),
    })
  },

  async unloadModel(name: string): Promise<{ success: boolean; message: string }> {
    return fetchApi<{ success: boolean; message: string }>(`/api/models/${encodeURIComponent(name)}/unload`, {
      method: 'POST',
    })
  },

  async listDefaults(): Promise<DefaultModel[]> {
    const response = await fetchApi<DefaultsResponse>('/api/defaults')
    return response.models
  },

  async useDefault(name: string): Promise<{ success: boolean; message: string; model?: any }> {
    return fetchApi<{ success: boolean; message: string; model?: any }>(`/api/defaults/${encodeURIComponent(name)}/use`, {
      method: 'POST',
    })
  },
}

// Prometheus metrics
export const metrics = {
  async get(): Promise<string> {
    const response = await fetch('/metrics')
    return response.text()
  },
}
