import { test, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { vi } from 'vitest'
import userEvent from '@testing-library/user-event'
import PredictorForm from '../components/PredictorForm'


test('renders PredictorForm component', () => {
  const mockOnSubmit = vi.fn()
  render(<PredictorForm model="sms" maxTextLen={160} loading={false} onSubmit={mockOnSubmit} />)

  const textarea = screen.getByPlaceholderText(/Paste a single SMS or email body…/i)
  expect(textarea).toBeInTheDocument()
})

test('confirm text does not go over maxTextLen', async () => {
  const onSubmit = vi.fn()
  const maxLen = 10
  render(<PredictorForm model="sms" maxTextLen={maxLen} loading={false} onSubmit={onSubmit} />)

  
  const textarea = screen.getByLabelText(/message/i)
  const user = userEvent.setup()
  const longText = 'a'.repeat(maxLen + 5)

  await user.type(textarea, longText)

  expect((textarea as HTMLTextAreaElement).value.length).toBe(maxLen + 5)

  expect(await screen.findByText(/too long/i)).toBeInTheDocument()

  const predictBtn = screen.getByRole('button', { name: /predict/i })
  expect(predictBtn).toBeDisabled()
})

test('disables submit button when loading', () => {
  const mockOnSubmit = vi.fn()
  render(<PredictorForm model="sms" maxTextLen={160} loading={true} onSubmit={mockOnSubmit} />)
  const predictBtn = screen.getByRole('button', { name: /predict/i })
  expect(predictBtn).toBeDisabled()
})





