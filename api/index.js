/**
 * Configuration (Environment Variables)
 * API_KEYS: Comma-separated Gemini API keys for upstream rotation.
 * MASTER_KEY: Optional custom Authorization Bearer token. Defaults to first API_KEY.
 */
const RAW_API_KEYS = (typeof process !== 'undefined' && process.env && process.env.API_KEYS) || '';
const API_KEYS = RAW_API_KEYS.split(',').map(k => k.trim()).filter(Boolean);
const MASTER_KEY = (typeof process !== 'undefined' && process.env && process.env.MASTER_KEY) || API_KEYS[0] || '';

if (!API_KEYS.length || !MASTER_KEY) {
  console.error('FATAL: API_KEYS or MASTER_KEY is not set. The worker will not function correctly.');
}

const TARGET_HOST = "generativelanguage.googleapis.com";

// ---------- Utility Functions ----------

async function getHash(text) {
  const encoder = new TextEncoder();
  const data = encoder.encode(text);
  return await crypto.subtle.digest('SHA-256', data);
}

function timingSafeEqual(a, b) {
  if (a.byteLength !== b.byteLength) return false;
  const ua = new Uint8Array(a);
  const ub = new Uint8Array(b);
  let result = 0;
  for (let i = 0; i < ua.length; i++) {
    result |= ua[i] ^ ub[i];
  }
  return result === 0;
}

let MASTER_HASH_CACHE = null;

function generateId() {
  return 'chatcmpl-' + crypto.randomUUID();
}

function mapFinishReason(finish) {
  const map = {
    STOP: 'stop',
    MAX_TOKENS: 'length',
    SAFETY: 'content_filter',
    RECITATION: 'content_filter'
  };
  return finish ? (map[finish] || finish.toLowerCase()) : null;
}

// 将 Gemini 模型列表转换为 OpenAI 格式
function convertModelListToOpenAI(geminiJson) {
  const models = geminiJson.models || [];
  const data = models.map(model => {
    const id = model.name.split('/').pop();
    return {
      id,
      object: 'model',
      created: model.updateTime ? Math.floor(new Date(model.updateTime).getTime() / 1000) : 1700000000,
      owned_by: 'google',
    };
  });
  return {
    object: 'list',
    data,
  };
}

// 从 Google API 获取完整模型列表（单页最大支持 1000 避免默认 50 个的分页截断）
async function fetchOfficialModelList() {
  const urls = API_KEYS.map(
    key => `https://${TARGET_HOST}/v1beta/models?key=${encodeURIComponent(key)}&pageSize=1000`
  );
  let lastError;

  for (const url of urls) {
    try {
      const resp = await fetch(url);
      if (resp.ok) {
        const json = await resp.json();
        return json;
      }
      lastError = new Error(`Upstream returned ${resp.status}`);
    } catch (e) {
      lastError = e;
    }
  }

  throw lastError || new Error('Failed to fetch model list from all keys');
}

// OpenAI 请求体转换为 Gemini 请求体
function convertBody(openaiBody) {
  const { model = 'gemini-pro', messages, temperature, top_p, max_tokens, stop, stream } = openaiBody ?? {};
  if (!messages || !Array.isArray(messages)) throw new Error('Missing messages array');

  const contents = [];
  let systemInstructionParts;

  for (const msg of messages) {
    if (msg.role === 'system') {
      if (!systemInstructionParts) systemInstructionParts = [];
      systemInstructionParts.push({ text: msg.content });
    } else if (msg.role === 'user' || msg.role === 'assistant') {
      contents.push({
        role: msg.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: msg.content }],
      });
    }
  }

  const generationConfig = {};
  if (temperature !== undefined) generationConfig.temperature = temperature;
  if (top_p !== undefined) generationConfig.topP = top_p;
  if (max_tokens !== undefined) generationConfig.maxOutputTokens = max_tokens;
  if (stop !== undefined) generationConfig.stopSequences = Array.isArray(stop) ? stop : [stop];

  const geminiBody = { contents, generationConfig };
  if (systemInstructionParts?.length) {
    geminiBody.systemInstruction = { parts: systemInstructionParts };
  }
  return { geminiBody, model, stream: stream || false };
}

// 非流式响应转换
function convertNonStreamResponse(geminiJson, requestModel) {
  const candidates = geminiJson.candidates || [];
  const choices = candidates.map((cand, idx) => ({
    index: idx,
    message: { role: 'assistant', content: cand.content?.parts?.[0]?.text || '' },
    finish_reason: mapFinishReason(cand.finishReason),
  }));

  return {
    id: generateId(),
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model: requestModel,
    usage: geminiJson.usageMetadata
      ? {
          prompt_tokens: geminiJson.usageMetadata.promptTokenCount || 0,
          completion_tokens: geminiJson.usageMetadata.candidatesTokenCount || 0,
          total_tokens:
            (geminiJson.usageMetadata.promptTokenCount || 0) +
            (geminiJson.usageMetadata.candidatesTokenCount || 0),
        }
      : undefined,
    choices,
  };
}

// 流式响应转换器
async function* streamConverter(reader, requestModel) {
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';
    for (const line of lines) {
      let trimmed = line.trim();
      if (!trimmed) continue;

      // 剥离 Google SSE 流自带的 "data:" 前缀
      if (trimmed.startsWith('data:')) {
        trimmed = trimmed.replace(/^data:\s*/, '');
      }
      trimmed = trimmed.trim();

      if (!trimmed || trimmed === '[DONE]' || trimmed.startsWith('[') || trimmed.startsWith(']')) continue;
      try {
        const geminiChunk = JSON.parse(trimmed);
        const candidates = geminiChunk.candidates || [];
        if (!candidates.length) continue;
        const deltaContent = candidates[0].content?.parts?.[0]?.text || '';
        const finishReason = candidates[0].finishReason;
        yield `data: ${JSON.stringify({
          id: generateId(),
          object: 'chat.completion.chunk',
          created: Math.floor(Date.now() / 1000),
          model: requestModel,
          choices: [{
            index: 0,
            delta: { role: 'assistant', content: deltaContent },
            finish_reason: mapFinishReason(finishReason),
          }],
        })}\n\n`;
      } catch {}
    }
  }
  yield 'data: [DONE]\n\n';
}

// 验证 Master Key
async function verifyMasterKey(clientToken) {
  if (!MASTER_HASH_CACHE) {
    MASTER_HASH_CACHE = await getHash(MASTER_KEY);
  }
  const clientHash = await getHash(clientToken);
  return timingSafeEqual(clientHash, MASTER_HASH_CACHE);
}

// ---------- Main Handler ----------

export default async function handler(request) {
  if (!API_KEYS.length || !MASTER_KEY) {
    return new Response(JSON.stringify({
      error: { message: 'Server misconfigured: API_KEYS or MASTER_KEY missing', type: 'internal_error', param: null, code: 'server_error' }
    }), {
      status: 500,
      headers: {
        'Content-Type': 'application/json; charset=utf-8',
        'Access-Control-Allow-Origin': '*',
      },
    });
  }

  // CORS 预检请求处理
  if (request.method === 'OPTIONS') {
    return new Response(null, {
      status: 204,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      },
    });
  }

  // 路径解析与重写兼容
  const url = new URL(request.url);
  let path = url.pathname;
  const realPath = request.headers.get('x-original-url') || 
                   request.headers.get('x-forwarded-path') || 
                   path;
  const v1Index = realPath.indexOf('/v1');
  if (v1Index !== -1) {
    path = realPath.substring(v1Index);
  }

  // 健康检查
  if (request.method === 'GET' && path === '/v1/health') {
    return new Response('OK', {
      status: 200,
      headers: { 'Access-Control-Allow-Origin': '*' },
    });
  }

  // 模型列表接口
  if (request.method === 'GET' && path === '/v1/models') {
    const authHeader = request.headers.get('Authorization');
    if (!authHeader?.startsWith('Bearer ')) {
      return new Response(JSON.stringify({
        error: { message: 'Missing or invalid Authorization header', type: 'invalid_api_key', param: null, code: 'invalid_api_key' }
      }), {
        status: 401,
        headers: { 'Content-Type': 'application/json; charset=utf-8', 'Access-Control-Allow-Origin': '*' },
      });
    }

    const clientToken = authHeader.slice(7).trim();
    try {
      if (!await verifyMasterKey(clientToken)) {
        return new Response(JSON.stringify({
          error: { message: 'Incorrect API key provided.', type: 'invalid_api_key', param: null, code: 'invalid_api_key' }
        }), {
          status: 401,
          headers: { 'Content-Type': 'application/json; charset=utf-8', 'Access-Control-Allow-Origin': '*' },
        });
      }
    } catch (e) {
      return new Response(JSON.stringify({
        error: { message: 'Authentication processing failed', type: 'internal_error', param: null, code: 'internal_error' }
      }), {
        status: 500,
        headers: { 'Content-Type': 'application/json; charset=utf-8', 'Access-Control-Allow-Origin': '*' },
      });
    }

    try {
      const geminiModelList = await fetchOfficialModelList();
      const openaiModelList = convertModelListToOpenAI(geminiModelList);
      return new Response(JSON.stringify(openaiModelList), {
        status: 200,
        headers: { 'Content-Type': 'application/json; charset=utf-8', 'Access-Control-Allow-Origin': '*' },
      });
    } catch (e) {
      return new Response(JSON.stringify({
        error: { message: `Failed to fetch upstream models: ${e.message}`, type: 'api_error', param: null, code: 'upstream_error' }
      }), {
        status: 502,
        headers: { 'Content-Type': 'application/json; charset=utf-8', 'Access-Control-Allow-Origin': '*' },
      });
    }
  }

  // 聊天补全接口校验
  if (path !== '/v1/chat/completions') {
    return new Response(JSON.stringify({
      error: { message: 'Not Found', type: 'invalid_request_error', param: null, code: 'not_found' }
    }), {
      status: 404,
      headers: { 'Content-Type': 'application/json; charset=utf-8', 'Access-Control-Allow-Origin': '*' },
    });
  }

  if (request.method !== 'POST') {
    return new Response(JSON.stringify({
      error: { message: 'Method not allowed', type: 'invalid_request_error', param: null, code: 'method_not_allowed' }
    }), {
      status: 405,
      headers: { 'Content-Type': 'application/json; charset=utf-8', 'Access-Control-Allow-Origin': '*' },
    });
  }

  // 主密钥鉴权
  const authHeader = request.headers.get('Authorization');
  if (!authHeader?.startsWith('Bearer ')) {
    return new Response(JSON.stringify({
      error: { message: 'Missing or invalid Authorization header', type: 'invalid_api_key', param: null, code: 'invalid_api_key' }
    }), {
      status: 401,
      headers: { 'Content-Type': 'application/json; charset=utf-8', 'Access-Control-Allow-Origin': '*' },
    });
  }

  const clientToken = authHeader.slice(7).trim();
  try {
    if (!await verifyMasterKey(clientToken)) {
      return new Response(JSON.stringify({
        error: { message: 'Incorrect API key provided.', type: 'invalid_api_key', param: null, code: 'invalid_api_key' }
      }), {
        status: 401,
        headers: { 'Content-Type': 'application/json; charset=utf-8', 'Access-Control-Allow-Origin': '*' },
      });
    }
  } catch (e) {
    return new Response(JSON.stringify({
      error: { message: 'Authentication processing failed', type: 'internal_error', param: null, code: 'internal_error' }
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json; charset=utf-8', 'Access-Control-Allow-Origin': '*' },
    });
  }

  // 解析请求体
  let requestBody;
  try {
    requestBody = await request.text();
  } catch {
    return new Response(JSON.stringify({
      error: { message: 'Invalid request body', type: 'invalid_request_error', param: null, code: 'invalid_body' }
    }), {
      status: 400,
      headers: { 'Content-Type': 'application/json; charset=utf-8', 'Access-Control-Allow-Origin': '*' },
    });
  }

  let parsed;
  try {
    parsed = JSON.parse(requestBody);
  } catch {
    return new Response(JSON.stringify({
      error: { message: 'Invalid JSON', type: 'invalid_request_error', param: null, code: 'json_parse_error' }
    }), {
      status: 400,
      headers: { 'Content-Type': 'application/json; charset=utf-8', 'Access-Control-Allow-Origin': '*' },
    });
  }

  let geminiBody, model, stream;
  try {
    const converted = convertBody(parsed);
    geminiBody = converted.geminiBody;
    model = converted.model;
    stream = converted.stream;
  } catch (e) {
    return new Response(JSON.stringify({
      error: { message: e.message, type: 'invalid_request_error', param: null, code: 'invalid_body' }
    }), {
      status: 400,
      headers: { 'Content-Type': 'application/json; charset=utf-8', 'Access-Control-Allow-Origin': '*' },
    });
  }

  // 拼接 Google 上游 URL
  const upstreamPath = stream
    ? `/v1beta/models/${model}:streamGenerateContent?alt=sse`
    : `/v1beta/models/${model}:generateContent`;
  const upstreamUrl = `https://${TARGET_HOST}${upstreamPath}`;

  let keyPool = [...API_KEYS];
  let lastResponse = null;

  // 密钥轮询与重试机制
  while (keyPool.length > 0) {
    const randomIndex = Math.floor(Math.random() * keyPool.length);
    const currentKey = keyPool.splice(randomIndex, 1)[0];

    const headers = new Headers();
    headers.set('Authorization', `Bearer ${currentKey}`);
    headers.set('Host', TARGET_HOST);
    headers.set('Content-Type', 'application/json');

    try {
      const response = await fetch(upstreamUrl, {
        method: 'POST',
        headers,
        body: JSON.stringify(geminiBody),
      });

      // 遇到限流或服务端错误，尝试使用密钥池内的下一个 Key
      if ((response.status === 429 || response.status >= 500) && keyPool.length > 0) {
        lastResponse = response;
        continue;
      }

      // 非流式响应处理
      if (!stream) {
        const geminiJson = await response.json();
        const openaiResponse = convertNonStreamResponse(geminiJson, model);
        return new Response(JSON.stringify(openaiResponse), {
          status: 200,
          headers: {
            'Content-Type': 'application/json; charset=utf-8',
            'Access-Control-Allow-Origin': '*',
          },
        });
      }

      // 流式响应处理
      if (!response.body) {
        return new Response(JSON.stringify({
          error: { message: 'Empty upstream stream', type: 'api_error', param: null, code: 'empty_stream' }
        }), {
          status: 502,
          headers: { 'Content-Type': 'application/json; charset=utf-8', 'Access-Control-Allow-Origin': '*' },
        });
      }

      const reader = response.body.getReader();
      const openaiStream = new ReadableStream({
        async start(controller) {
          const it = streamConverter(reader, model);
          try {
            for await (const chunk of it) {
              controller.enqueue(new TextEncoder().encode(chunk));
            }
            controller.close();
          } catch (err) {
            controller.error(err);
          }
        },
      });

      return new Response(openaiStream, {
        status: 200,
        headers: {
          'Content-Type': 'text/event-stream; charset=utf-8',
          'Cache-Control': 'no-cache, no-transform',
          'Access-Control-Allow-Origin': '*',
        },
      });

    } catch (err) {
      if (keyPool.length > 0) continue;
      return new Response(JSON.stringify({
        error: { message: `Upstream error: ${err.message}`, type: 'api_error', param: null, code: 'bad_gateway' }
      }), {
        status: 502,
        headers: { 'Content-Type': 'application/json; charset=utf-8', 'Access-Control-Allow-Origin': '*' },
      });
    }
  }

  // 密钥全耗尽或均被限流时的安全错误返回
  if (lastResponse) {
    return new Response(JSON.stringify({
      error: { 
        message: `Upstream error. Last status code: ${lastResponse.status}. All API keys exhausted or rate limited.`, 
        type: 'api_error', 
        param: null, 
        code: 'key_exhausted' 
      }
    }), {
      status: lastResponse.status,
      headers: {
        'Content-Type': 'application/json; charset=utf-8',
        'Access-Control-Allow-Origin': '*',
      },
    });
  }

  return new Response(JSON.stringify({
    error: { message: 'All API keys exhausted / rate limited', type: 'api_error', param: null, code: 'key_exhausted' }
  }), {
    status: 503,
    headers: { 'Content-Type': 'application/json; charset=utf-8', 'Access-Control-Allow-Origin': '*' },
  });
}

export const config = {
  runtime: 'edge',
};
