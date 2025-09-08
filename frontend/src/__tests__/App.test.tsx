import { render, screen, within  } from '@testing-library/react'
import { test, expect } from 'vitest'
import App from '../App'
import { server } from '../test-utils/msw/server'
import { http, HttpResponse } from 'msw'


test('shows model names after loading /meta', async () => {
    render(<App />)

    // const sms = await screen.findByText(/sms/i)
    // const email = await screen.findByText(/email/i)

    // expect(sms).toBeInTheDocument()
    // expect(email).toBeInTheDocument()
    const modelSelect = await screen.findByLabelText(/model/i)
    expect(modelSelect).toBeInTheDocument()

    const optSms = within(modelSelect).getByRole('option', { name: /sms/i })
    const optEmail = within(modelSelect).getByRole('option', { name: /email/i })

    expect(optSms).toBeInTheDocument()
    expect(optEmail).toBeInTheDocument()
})


test('shows an error message when /meta fails', async () => {
   server.resetHandlers(
    http.get('http://127.0.0.1:8000/meta', () =>
      HttpResponse.json({ detail: 'boom' }, { status: 500 })
    ),
    http.get('http://127.0.0.1:8000/health', () =>
      HttpResponse.json({ status: 'ok' })
    )
  )

  render(<App />)

  expect(await screen.findByText(/offline/i)).toBeInTheDocument()
  expect(await screen.findByLabelText(/api status: offline/i)).toBeInTheDocument()
})
