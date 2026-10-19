export type Block =
  | { type: "heading" | "paragraph"; text: string }
  | { type: "table"; caption: string; headers: string[]; rows: string[][] };

export type SourceDoc = { id: string; file: string; title: string; version: string; updatedAt: string; blocks: Block[] };

/** A small, fictional product knowledge base. Each document exists to make a specific failure observable. */
export const CORPUS: SourceDoc[] = [
  {
    id: "pricing", file: "pricing.pdf", title: "Plans and pricing", version: "v3", updatedAt: "2026-09-01",
    blocks: [
      { type: "heading", text: "Plans and pricing" },
      { type: "paragraph", text: "Northwind Cloud offers three plans. Every plan includes unlimited projects and the web dashboard." },
      { type: "table", caption: "Plan comparison", headers: ["Plan", "Price", "Seats", "Support"], rows: [["Starter", "$0 per month", "1", "community forum"], ["Pro", "$49 per month", "10", "email"], ["Enterprise", "custom pricing", "unlimited", "24/7 phone"]] },
      { type: "paragraph", text: "Annual billing gives you two months free on any paid plan." },
    ],
  },
  {
    id: "refunds", file: "refund-policy.md", title: "Refund policy", version: "v2", updatedAt: "2026-06-12",
    blocks: [
      { type: "heading", text: "Refund policy" },
      { type: "paragraph", text: "Northwind Cloud bills monthly or annually. This page explains when a charge can be refunded and how to request one through the billing portal or by contacting the support team." },
      { type: "paragraph", text: "Monthly plans can be refunded within 14 days of the charge if usage stayed under the free tier limits." },
      { type: "paragraph", text: "Annual plans can be refunded within 30 days of purchase, prorated for any usage above the free tier." },
      { type: "paragraph", text: "After the refund window closes you can still cancel renewal from the billing portal." },
    ],
  },
  {
    id: "sso", file: "sso-setup.md", title: "Single sign-on setup", version: "v1", updatedAt: "2026-03-30",
    blocks: [
      { type: "heading", text: "Single sign-on setup" },
      { type: "paragraph", text: "Single sign-on lets your team log in with your company identity provider. SSO is available on the Enterprise plan only." },
      { type: "paragraph", text: "To enable SSO an organization admin opens Settings then Security then Single sign-on and uploads the identity provider metadata XML file." },
      { type: "paragraph", text: "Finally the admin verifies the company domain by adding a DNS TXT record." },
    ],
  },
  {
    id: "retention", file: "data-retention.md", title: "Data retention", version: "v1", updatedAt: "2026-05-02",
    blocks: [
      { type: "heading", text: "Data retention" },
      { type: "paragraph", text: "Request logs are retained for 30 days on Starter and Pro plans and for 365 days on the Enterprise plan." },
      { type: "paragraph", text: "Deleted projects can be recovered for 7 days before they are permanently removed." },
    ],
  },
  {
    id: "rate-limits", file: "api-rate-limits.md", title: "API rate limits", version: "v3", updatedAt: "2026-10-01",
    blocks: [
      { type: "heading", text: "API rate limits" },
      { type: "paragraph", text: "The Pro plan allows 600 requests per minute per API key. The Starter plan allows 60 requests per minute." },
      { type: "paragraph", text: "Requests above the limit receive HTTP 429 with a Retry-After header that says how many seconds to wait." },
    ],
  },
];

/** The previous published version of the rate-limit page. A stale index still contains this. */
export const RATE_LIMITS_V2: SourceDoc = {
  ...CORPUS[4], version: "v2", updatedAt: "2026-04-14",
  blocks: [
    { type: "heading", text: "API rate limits" },
    { type: "paragraph", text: "The Pro plan allows 300 requests per minute per API key. The Starter plan allows 60 requests per minute." },
    { type: "paragraph", text: "Requests above the limit receive HTTP 429 with a Retry-After header that says how many seconds to wait." },
  ],
};

export type Question = { id: string; text: string; expect: string; sourceDoc: string };

/** Golden set: each question has one expected fact and the document that should supply it. */
export const QUESTIONS: Question[] = [
  { id: "q-price", text: "How much does the Pro plan cost per month?", expect: "$49", sourceDoc: "pricing" },
  { id: "q-refund", text: "Can I get a refund on an annual plan after three weeks?", expect: "30 days of purchase", sourceDoc: "refunds" },
  { id: "q-sso", text: "How do I enable SSO?", expect: "metadata XML", sourceDoc: "sso" },
  { id: "q-retention", text: "How long are request logs kept on the Enterprise plan?", expect: "365 days", sourceDoc: "retention" },
  { id: "q-rate", text: "What is the API rate limit on the Pro plan?", expect: "600 requests per minute", sourceDoc: "rate-limits" },
  { id: "q-429", text: "What happens when I exceed the rate limit?", expect: "429", sourceDoc: "rate-limits" },
];

/**
 * The simulated generator's "parametric memory": facts it would state without context.
 * Deliberately outdated, so ungrounded answers are detectably wrong.
 */
export const MODEL_PRIOR: Record<string, string> = {
  "q-rate": "The Pro plan allows 300 requests per minute.",
  "q-price": "The Pro plan costs $39 per month.",
};
