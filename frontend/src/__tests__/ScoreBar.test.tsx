import { test, expect } from 'vitest'
import { render, screen} from '@testing-library/react'
import ScoreBar from '../components/ScoreBar'

test('renders 75% with correct aria attributes', () => {
  render(<ScoreBar value={0.75} />)

  const bar = screen.getByRole('progressbar', { name: /spam score/i })
  expect(bar).toHaveAttribute('aria-valuemin', '0')
  expect(bar).toHaveAttribute('aria-valuemax', '100')
  expect(bar).toHaveAttribute('aria-valuenow', '75')

  expect(screen.getByText('75%')).toBeInTheDocument()

  const filler = bar.firstElementChild as HTMLElement | null
  expect(filler).not.toBeNull()
  expect(filler!).toHaveStyle({ width: '75%' })
})

test('clamps negative score to 0%', () => {
  render(<ScoreBar value={-1} />)
  const bar = screen.getByRole('progressbar', { name: /spam score/i })
  expect(bar).toHaveAttribute('aria-valuenow', '0')
  expect(screen.getByText('0%')).toBeInTheDocument()
})


test('clamps score over 1 to 100%', () => {
  render(<ScoreBar value={1.5} />)
  const bar = screen.getByRole('progressbar', { name: /spam score/i })
  expect(bar).toHaveAttribute('aria-valuenow', '100')
  expect(screen.getByText('100%')).toBeInTheDocument()
})