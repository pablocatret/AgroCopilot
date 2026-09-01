import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import test from "node:test"

test("the product demo is decoupled from legacy delivery assets", () => {
  const app = readFileSync(new URL("../App.tsx", import.meta.url), "utf8")
  const fixture = readFileSync(new URL("./productDemo.ts", import.meta.url), "utf8")

  assert.match(app, /from "\.\/demo\/productDemo"/)
  assert.match(app, /get\("demo"\) === "product"/)
  const prohibitedLegacyLabel = new RegExp(["the", "sis"].join(""), "i")
  assert.doesNotMatch(`${app}\n${fixture}`, prohibitedLegacyLabel)
})
