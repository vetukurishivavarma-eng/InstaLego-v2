"""Thin HTTP delivery layer over the landtitle pipeline.

Nothing under src/landtitle/ (the pipeline itself) is modified by this
package -- it only imports and calls into it. See jobs.py for the async
job model and app.py for the FastAPI routes.
"""
