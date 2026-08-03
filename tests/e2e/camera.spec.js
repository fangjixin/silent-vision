const { test, expect } = require("@playwright/test");

test("home page exposes camera controls and status", async ({ page }) => {
  await page.goto("http://127.0.0.1:8000/");
  await expect(page.locator("#startButton")).toBeVisible();
  await expect(page.locator("#cameraStatus")).toContainText("idle");
  await expect(page.locator("#bufferStatus")).toContainText("waiting");
});
