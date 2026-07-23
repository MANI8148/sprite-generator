import { render, screen } from "@testing-library/react";
import Layout from "../components/Layout";

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

describe("Layout", () => {
  it("renders children", () => {
    render(
      <Layout>
        <div data-testid="child">Hello</div>
      </Layout>
    );
    expect(screen.getByTestId("child")).toBeTruthy();
    expect(screen.getByText("Hello")).toBeTruthy();
  });

  it("renders Navbar", () => {
    render(
      <Layout>
        <div>Content</div>
      </Layout>
    );
    expect(screen.getByText("Generate")).toBeTruthy();
  });

  it("renders Billing in navbar", () => {
    render(
      <Layout>
        <div>Content</div>
      </Layout>
    );
    expect(screen.getByText("Billing")).toBeTruthy();
  });
});
