import { Link } from "react-router";

export function NotFound() {
  return (
    <div className="mx-auto max-w-md py-16 text-center">
      <h1 className="text-xl font-semibold">Page not found</h1>
      <p className="mt-2 text-sm text-muted">That topic or page does not exist.</p>
      <Link to="/explore" className="mt-4 inline-block text-sm text-accent hover:underline">Go to Explore</Link>
    </div>
  );
}
