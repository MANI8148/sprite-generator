import Link from "next/link";
import { useRouter } from "next/router";

const links = [
  { href: "/", label: "Generate" },
  { href: "/history", label: "History" },
  { href: "/downloads", label: "Downloads" },
  { href: "/settings", label: "Settings" },
];

export default function Navbar() {
  const router = useRouter();

  return (
    <nav
      style={{
        background: "#1a1a2e",
        padding: "1rem 2rem",
        display: "flex",
        gap: "1.5rem",
        borderBottom: "1px solid #333",
      }}
    >
      <span style={{ fontWeight: "bold", color: "#7c7cff", marginRight: "auto" }}>
        Sprite Generator
      </span>
      {links.map((link) => (
        <Link
          key={link.href}
          href={link.href}
          style={{
            color: router.pathname === link.href ? "#fff" : "#999",
            fontWeight: router.pathname === link.href ? "bold" : "normal",
          }}
        >
          {link.label}
        </Link>
      ))}
    </nav>
  );
}
