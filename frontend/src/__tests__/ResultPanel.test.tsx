import { test, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import ResultPanel from '../components/ResultPanel'
import { within } from '@testing-library/react'



test('Renders placeholder text when no result is provided', () => {
  const { getByText } = render(<ResultPanel model="sms" result={null} />)
  expect(getByText('Paste a message and click Predict to see results')).toBeInTheDocument()
})

test('Displays model name and applies uppercase style', () => {
  render(<ResultPanel model="sms" result={{ label: 'spam', score: 0.90 }} />)

  const modelLine = screen.getByText(/model:/i).closest('p')
  expect(modelLine).toBeInTheDocument()

  const modelName = within(modelLine as HTMLElement).getByText(/sms/i)
  expect(modelName).toBeInTheDocument()

  expect(modelName).toHaveClass('uppercase')
})

test('Displays spam prediction in red', () => {
  const { getByText } = render(<ResultPanel model="sms" result={{ label: 'spam' }} />)
  const prediction = getByText('spam')
  expect(prediction).toBeInTheDocument()
  expect(prediction).toHaveClass('text-red-600')
})

test('Displays ham prediction in green', () => {
  const { getByText } = render(<ResultPanel model="sms" result={{ label: 'ham' }} />)
  const prediction = getByText('ham')
  expect(prediction).toBeInTheDocument()
  expect(prediction).toHaveClass('text-green-600')
})

test('Shows spam score percentage', () => {
  render(<ResultPanel model="sms" result={{ label: 'spam', score: 0.75 }} />)

  const bar = screen.getByRole('progressbar', { name: /spam score/i })
  expect(bar).toHaveAttribute('aria-valuenow', '75')

  expect(screen.getByText('75%')).toBeInTheDocument()
})




