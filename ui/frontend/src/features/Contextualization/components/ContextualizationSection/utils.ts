/*highlight searched text from contextualization filters*/
export function highlight(text: string, query: string) {
  if (!query) return text;

  const safe = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

  const regex = new RegExp(`(${safe})`, 'gi');
  return text.replace(regex, `<span class="highlighted">$1</span>`);
}
