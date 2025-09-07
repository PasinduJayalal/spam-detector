import { render, screen } from '@testing-library/react'
import { test, expect } from 'vitest'
import App from '../App'


test('shows model names after loading /meta', async () => {
    render(<App />)

    const sms = await screen.findByText(/sms/i)
    const email = await screen.findByText(/email/i)

    expect(sms).toBeInTheDocument()
    expect(email).toBeInTheDocument()
})


test('shows an error message when /meta fails', async () => {
    
    const { server } = await import('../test-utils/msw/server')
    const { http, HttpResponse } = await import('msw')

    server.resetHandlers(
        http.get('/meta', () => HttpResponse.json({ detail: 'boom' }, { status: 500 }))
    )

    render(<App />)

    const errorMsg = await screen.findByText(/error/i)
    expect(errorMsg).toBeInTheDocument()
})
