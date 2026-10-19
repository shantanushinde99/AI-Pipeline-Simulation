import js from "@eslint/js";
import globals from "globals";
import reactHooks from "eslint-plugin-react-hooks";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist", "playwright-report", "test-results"] },
  {
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    files: ["**/*.{ts,tsx}"],
    languageOptions: { ecmaVersion: 2022, globals: globals.browser },
  },
  // Hooks rules only for app code: Playwright fixtures also call a function named `use`.
  { files: ["src/**/*.{ts,tsx}"], plugins: { "react-hooks": reactHooks }, rules: { ...reactHooks.configs.recommended.rules } },
);
