import { http, HttpResponse } from 'msw'

export const handlers = [
  
  http.get('/meta', () => {
    return HttpResponse.json({
      models: ['sms', 'email'],
      max_text_len: 5000,
    })
  }),
  
]