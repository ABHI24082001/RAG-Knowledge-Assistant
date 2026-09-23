export const NO_ANSWER = 'I could not find that information in the indexed documents.'

export function buildChatPayload(question, selectedDocumentId) {
  const payload = {
    question: question.trim(),
    top_k: 3,
  }

  // The select stores document.document_id as its value. The filename is only
  // presentation text and must never be sent as document_id.
  if (selectedDocumentId) payload.document_id = selectedDocumentId

  return payload
}
