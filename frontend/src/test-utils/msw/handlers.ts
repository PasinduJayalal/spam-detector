import { http, HttpResponse } from 'msw'

const API = 'http://127.0.0.1:8000'


export const handlers = [

  http.get(`${API}/health`, () => {
    return HttpResponse.json({ status: 'ok' })
  }),
  
  http.get('${API}/meta', () => {
    return HttpResponse.json({
      models: ['sms', 'email'],
      max_text_len: 4000,
    })
  }),
  http.post(`${API}/predict`, async ({ request }) => {
    const { model, text } = (await request.json()) as {
      model?: string
      text?: string
    }

    const t = (text ?? '').toLowerCase().trim()

    if (!t) {
      return HttpResponse.json(
        { detail: 'text must not be empty' },
        { status: 400 }
      )
    }

    const isSpam = /win|free|prize|claim/.test(t)

    return HttpResponse.json({
      model: model ?? 'sms',
      label: isSpam ? 'Spam' : 'Not Spam',
      score: isSpam ? 0.88 : 0.12,
    })
  }),
]