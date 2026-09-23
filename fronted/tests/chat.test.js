import assert from 'node:assert/strict'
import test from 'node:test'

import { buildChatPayload } from '../src/utils/chat.js'

test('selected document payload contains its UUID', () => {
  const documentId = '6b4d9649-68a7-4df5-bb4e-0a47567680d8'

  assert.deepEqual(
    buildChatPayload('How much experience does Abhishek Kumar have?', documentId),
    {
      question: 'How much experience does Abhishek Kumar have?',
      top_k: 3,
      document_id: documentId,
    },
  )
})

test('all-documents payload omits document_id', () => {
  assert.deepEqual(
    buildChatPayload('Summarize the documents', ''),
    {
      question: 'Summarize the documents',
      top_k: 3,
    },
  )
})
