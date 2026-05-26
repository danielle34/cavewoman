# Deployment

Plain static HTML, CDN-hosted Bulma + Font Awesome + Academic Icons. No
build step.

## GitHub Pages (gh-pages branch)

```bash
git checkout --orphan gh-pages && git rm -rf .
cp -R site/. . && git add . && git commit -m "site"
git push origin gh-pages
```

Settings → Pages → Source: branch `gh-pages`, root.

## GitHub Pages (/docs on main)

```bash
mkdir -p docs && cp -R site/. docs/
git add docs && git commit -m "site" && git push
```

Settings → Pages → Source: branch `main`, folder `/docs`.

## Any static host

Drop `site/` contents into Cloudflare Pages, Netlify, Vercel, or S3.

## CDN deps

Bulma 0.9.4, bulma-carousel 4.0.4, Font Awesome 5.15.4, Academic Icons,
jQuery 3.5.1, Google Fonts. Mirror into `static/vendor/` for offline.
