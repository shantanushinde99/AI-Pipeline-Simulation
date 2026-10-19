import { useState, type FormEvent } from "react";
import { Link, NavLink, Outlet, useNavigate } from "react-router";
import { BookOpen, Compass, FlaskConical, GraduationCap, Menu, Moon, Search, Sun, Trophy, X } from "lucide-react";
import { cx } from "../ui";

const NAV = [
  { to: "/explore", label: "Explore", Icon: Compass },
  { to: "/paths", label: "Learning Paths", Icon: BookOpen },
  { to: "/failure-lab", label: "Failure Lab", Icon: FlaskConical },
  { to: "/practice", label: "Interview Practice", Icon: GraduationCap },
  { to: "/progress", label: "Progress", Icon: Trophy },
];

function useTheme() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains("dark"));
  const toggle = () => {
    const next = !dark;
    document.documentElement.classList.toggle("dark", next);
    try { localStorage.setItem("asv-theme", next ? "dark" : "light"); } catch { /* ignore */ }
    setDark(next);
  };
  return { dark, toggle };
}

export function Logo() {
  return (
    <Link to="/explore" className="flex shrink-0 items-center gap-2 font-semibold tracking-tight">
      <img src="/favicon.svg" alt="" width={24} height={24} />
      <span>AI Systems Visualizer</span>
    </Link>
  );
}

export function AppShell() {
  const { dark, toggle } = useTheme();
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const navigate = useNavigate();
  const submit = (e: FormEvent) => {
    e.preventDefault();
    navigate(`/explore?q=${encodeURIComponent(q.trim())}`);
    setOpen(false);
  };

  const links = (
    <>
      {NAV.map(({ to, label, Icon }) => (
        <NavLink
          key={to}
          to={to}
          onClick={() => setOpen(false)}
          className={({ isActive }) => cx("flex items-center gap-2 rounded-md px-2.5 py-1.5 text-sm font-medium transition-colors", isActive ? "bg-accent-soft text-accent" : "text-muted hover:bg-surface-2 hover:text-fg")}
        >
          <Icon size={15} aria-hidden />
          {label}
        </NavLink>
      ))}
    </>
  );

  const search = (
    <form onSubmit={submit} role="search" className="relative">
      <Search size={14} className="pointer-events-none absolute top-1/2 left-2.5 -translate-y-1/2 text-muted" aria-hidden />
      <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search topics…" aria-label="Search topics"
        className="h-8 w-full rounded-md border border-line bg-surface-2 pr-2 pl-8 text-sm placeholder:text-muted lg:w-52" />
    </form>
  );

  return (
    <div className="flex min-h-full flex-col">
      <header className="sticky top-0 z-40 border-b border-line bg-surface/95 backdrop-blur">
        <div className="mx-auto flex h-13 max-w-[1600px] items-center gap-3 px-4">
          <Logo />
          <nav aria-label="Primary" className="ml-4 hidden items-center gap-0.5 lg:flex">{links}</nav>
          <div className="ml-auto hidden lg:block">{search}</div>
          <button onClick={toggle} className="ml-auto rounded-md p-2 text-muted hover:bg-surface-2 hover:text-fg lg:ml-0" aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}>
            {dark ? <Sun size={16} /> : <Moon size={16} />}
          </button>
          <button onClick={() => setOpen(true)} className="rounded-md p-2 text-muted hover:bg-surface-2 lg:hidden" aria-label="Open menu">
            <Menu size={18} />
          </button>
        </div>
      </header>
      {open && (
        <div className="fixed inset-0 z-50 lg:hidden" role="dialog" aria-modal="true" aria-label="Menu">
          <div className="absolute inset-0 bg-black/40" onClick={() => setOpen(false)} />
          <div className="absolute inset-y-0 right-0 flex w-72 max-w-[85vw] flex-col gap-3 bg-surface p-4 shadow-xl">
            <div className="flex items-center justify-between">
              <span className="font-semibold">Menu</span>
              <button onClick={() => setOpen(false)} aria-label="Close menu" className="rounded-md p-1.5 hover:bg-surface-2"><X size={18} /></button>
            </div>
            {search}
            <nav aria-label="Primary mobile" className="flex flex-col gap-0.5">{links}</nav>
          </div>
        </div>
      )}
      <main className="mx-auto w-full max-w-[1600px] flex-1 px-4 py-5">
        <Outlet />
      </main>
    </div>
  );
}
