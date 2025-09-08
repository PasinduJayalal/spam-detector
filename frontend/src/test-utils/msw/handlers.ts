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
  
]