# Land Title Diligence — Frontend (Vercel)

A minimal single-page Next.js (App Router) frontend for the `POST /generate-opinion`
endpoint of the FastAPI backend in this repo (`src/landtitle/api/`). It uploads a
Sale Deed (required, multiple files allowed), an optional Revenue Record, and an
optional Encumbrance Certificate, then shows the returned Legal Opinion PDF for
preview and download.

This app is meant to be deployed to Vercel's free tier, while the backend keeps
running elsewhere (e.g. on your own laptop, exposed publicly via ngrok).

## Architecture note — why the fetch is client-side only

The browser calls the backend's `/generate-opinion` endpoint **directly** using
`fetch()` inside a client component (`app/page.js`). There is no Next.js API
route (`app/api/.../route.js`) proxying this request, and there must never be
one added later.

Why: Vercel's free-tier serverless functions time out after roughly 10 seconds.
The backend pipeline (OCR + several sequential LLM calls) can take multiple
minutes. If the request were routed through a Vercel serverless function, that
function would time out long before the backend finishes, even though the
backend itself is perfectly capable of completing the job. Calling the backend
directly from the browser avoids Vercel's timeout entirely — the browser will
happily wait several minutes for a `fetch()` to resolve.

The tradeoff is that CORS must be handled on the backend (allowing the Vercel
origin, or `*`, to call it) — that is being handled by the backend effort in
this same repo, not by this frontend.

## Running locally

```bash
cd frontend-vercel
npm install
```

Create a `.env.local` file in `frontend-vercel/` (this file is gitignored and
will not be committed):

```
NEXT_PUBLIC_API_BASE_URL=https://your-current-tunnel.ngrok-free.app
```

Then start the dev server:

```bash
npm run dev
```

Open http://localhost:3000. Select a Sale Deed (or several), optionally a
Revenue Record and/or Encumbrance Certificate, and click Submit.

## Backend URL: build-time env var vs. runtime override

`NEXT_PUBLIC_API_BASE_URL` is the **default** backend URL, and it is baked into
the JavaScript bundle **at build time** (this is how Next.js's `NEXT_PUBLIC_*`
env vars work — they are not read at runtime from the server environment).

Because an ngrok tunnel URL can change (e.g. every time you restart the tunnel
on a free ngrok plan), the page also has a small **"Backend URL settings"**
disclosure below the form:

- It's pre-filled from `NEXT_PUBLIC_API_BASE_URL`.
- Editing it immediately updates the URL used for the next submission — no
  rebuild or redeploy required.
- The value is persisted to the browser's `localStorage`, so it survives a
  page reload.
- A "Reset to default" button clears the override and falls back to the
  build-time `NEXT_PUBLIC_API_BASE_URL` value again.

**In short:** if your tunnel restarts with a new URL, just paste the new URL
into the "Backend URL settings" field on the page — you do not need to touch
Vercel at all for a quick fix. Updating the `NEXT_PUBLIC_API_BASE_URL`
environment variable in Vercel's dashboard is only useful as the *new default*
for anyone loading the page for the first time (or after clearing
localStorage), and that change only takes effect after you click **Redeploy**
in Vercel, because the value is inlined at build time.

## Deploying to Vercel (step-by-step)

1. **Push this repo to GitHub/GitLab/Bitbucket** (or make sure your existing
   remote is up to date) so Vercel can access it. The whole repo can be pushed
   as-is — Vercel will be pointed at the `frontend-vercel` subdirectory in the
   next step, so the rest of the repo (Python backend, etc.) is simply
   ignored by this Vercel project.
2. Go to https://vercel.com and log in (create a free account if you haven't).
3. Click **Add New... → Project**, then import this repository.
4. On the **Configure Project** screen:
   - Set **Root Directory** to `frontend-vercel` (click "Edit" next to Root
     Directory and select/type it). This is required — without it, Vercel
     will try to build from the repo root and fail, since the Next.js app
     lives in this subdirectory.
   - Framework Preset should auto-detect as **Next.js**. Leave the build
     command/output settings at their defaults.
5. Expand **Environment Variables** and add:
   - Name: `NEXT_PUBLIC_API_BASE_URL`
   - Value: your current ngrok URL, e.g. `https://your-tunnel.ngrok-free.app`
     (no trailing slash needed)
6. Click **Deploy**. Wait for the build to finish; Vercel will give you a
   `*.vercel.app` URL.
7. Open that URL, try a submission. If your backend's ngrok URL ever changes
   later, you have two options:
   - **Quick / no redeploy:** open the deployed page, expand "Backend URL
     settings", paste the new URL. Takes effect immediately for that browser.
   - **Update the default for all future visitors:** go to your Vercel
     project → **Settings → Environment Variables**, edit
     `NEXT_PUBLIC_API_BASE_URL` to the new URL, then go to **Deployments** and
     click **Redeploy** on the latest deployment (editing the env var alone
     does *not* affect the already-built deployment).

## What this page does not do

- It does not talk to any other backend route besides `POST {backendUrl}/generate-opinion`.
- It does not proxy, cache, or transform the request/response in any Vercel
  server-side code — see the architecture note above.
- It does not attempt retries; a failed request simply surfaces the backend's
  `detail` message (or a network-error message if the backend was
  unreachable) and lets you resubmit.
