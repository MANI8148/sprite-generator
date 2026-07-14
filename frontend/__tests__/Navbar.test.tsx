import { render, screen } from "@testing-library/react";
import Navbar from "../components/Navbar";

jest.mock("next/router", () => ({
  useRouter: () => ({ pathname: "/" }),
}));

jest.mock("next/link", () => {
  const MockLink = ({ children, href, ...rest }: { children: React.ReactNode; href: string }) => (
    <a href={href} {...rest}>
      {children}
    </a>
  );
  MockLink.displayName = "MockLink";
  return MockLink;
});

describe("Navbar", () => {
  it("renders all navigation links", () => {
    render(<Navbar />);
    expect(screen.getByText("Generate")).toBeTruthy();
    expect(screen.getByText("History")).toBeTruthy();
    expect(screen.getByText("Downloads")).toBeTruthy();
    expect(screen.getByText("Settings")).toBeTruthy();
  });

  it("renders the brand name", () => {
    render(<Navbar />);
    expect(screen.getByText("Sprite Generator")).toBeTruthy();
  });
});
