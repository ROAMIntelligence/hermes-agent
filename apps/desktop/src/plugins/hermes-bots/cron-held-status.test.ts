/**
 * The inspector's "Last result" row is the only place a desktop user sees why a
 * scheduled run stopped, and `routineLastResult` used to pass an unknown literal
 * through verbatim — so a platform stop rendered the raw
 * `interrupted_after_output` while `hermes cron list` called the same run "not a
 * job failure". The scheduler books those two literals when the platform stops a
 * run it had already started (the failure streak is held, not advanced), so the
 * label must say "held", and it must never be the literal itself.
 *
 * The set is still open on the gateway side: an unrecognised literal keeps
 * passing through verbatim rather than disappearing behind a generic word.
 */

import { describe, expect, it } from 'vitest'

import { routineLastResult } from './cron'
import { translateBotsIn } from './i18n-test-helper'

const HELD = ['interrupted', 'interrupted_after_output'] as const

const ENGLISH: Record<(typeof HELD)[number], string> = {
  interrupted: 'Stopped by a platform shutdown (not a job failure)',
  interrupted_after_output:
    'Stopped by a platform shutdown after its output was written (not a job failure)',
}

describe('routineLastResult held platform statuses', () => {
  it('names both held literals instead of echoing them', () => {
    for (const status of HELD) {
      expect(routineLastResult(status)).toBe(ENGLISH[status])
      expect(routineLastResult(status)).not.toBe(status)
    }
  })

  it('carries a real translation for every shipped locale', () => {
    for (const locale of ['ja', 'zh', 'zh-hant'] as const) {
      const t = translateBotsIn(locale)
      for (const key of ['cron.interrupted', 'cron.interruptedAfterOutput'] as const) {
        const value = t(key)
        expect(value).not.toBe(key) // resolved, not the key fallback
        expect(value.length).toBeGreaterThan(8)
        expect(value).not.toBe(ENGLISH[key === 'cron.interrupted' ? 'interrupted' : 'interrupted_after_output'])
      }
    }
  })

  it('still passes an unknown literal through verbatim', () => {
    expect(routineLastResult('something_new')).toBe('something_new')
  })
})
