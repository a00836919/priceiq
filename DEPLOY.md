# 🚀 DEPLOY.md — Guía paso a paso de despliegue a Azure

Esta guía toma tu instalación local de PriceIQ y la lleva a Azure como un
servicio funcional, con CI/CD automático en cada push a `main`.

**Tiempo total:** ~3-4 horas la primera vez. Después, cada deploy es
automático en 5 min vía GitHub Actions.

**Costo estimado:** $5-15 USD/mes con scale-to-zero. Si tienes Azure Student
o el crédito gratis ($200), te dura meses.

---

## 0. Prerequisitos

```bash
# Instalar Azure CLI
brew install azure-cli

# Instalar GitHub CLI (para configurar secrets)
brew install gh

# Login
az login
gh auth login
```

Confirma tu suscripción:
```bash
az account show --query "{name:name, id:id}" -o table
```

Si tienes varias, selecciona la que vas a usar:
```bash
az account set --subscription "<tu-subscription-id>"
```

---

## 1. Día 2 — Provisión de Azure (semi-automático)

### 1.1 Editar variables si quieres customizar

Abre `deploy/01-azure-setup.sh` y revisa las vars al inicio:

```bash
LOCATION="eastus"            # región — cámbiala si prefieres mexicocentral
RG="priceiq-rg"              # resource group
SUFFIX=$RANDOM               # sufijo aleatorio para nombres únicos globales
```

### 1.2 Ejecutar el script

```bash
cd /Users/iguacio/Documents/priceIQ
./deploy/01-azure-setup.sh
```

El script tarda ~10 min y hace, por orden:

1. Crea Resource Group `priceiq-rg`
2. Crea Storage Account + File Share + sube `data/` y `models/`
3. Crea Container Registry y construye la imagen Docker (en Azure, no en tu Mac)
4. Crea Log Analytics + Application Insights
5. Crea Container Apps environment con el File Share mounted
6. Crea el Container App con las env vars correctas

### 1.3 Al final imprime:

```
✅ DEPLOY COMPLETO

Backend API:    https://priceiq-api.xxxxx.eastus.azurecontainerapps.io
Healthcheck:    https://priceiq-api.xxxxx.eastus.azurecontainerapps.io/api/health
Swagger docs:   https://priceiq-api.xxxxx.eastus.azurecontainerapps.io/docs

Guarda estos valores para los GitHub Actions secrets:
  AZURE_RG=priceiq-rg
  AZURE_ACR=priceiq12345
  AZURE_APP=priceiq-api
```

**Anota la URL pública del backend** — la vas a necesitar para el frontend.

### 1.4 Verificar manualmente

```bash
# Healthcheck
curl https://TU-URL/api/health

# Debería responder:
# {"status":"ok","ready":true,"n_skus":7301,...}
```

Si tarda en responder los primeros 60-90s es **normal** — el container hace cold start.

### 1.5 Ver logs en vivo

```bash
az containerapp logs show -n priceiq-api -g priceiq-rg --follow
```

---

## 2. Día 3 — Frontend en Static Web Apps

### 2.1 Crear el Static Web App

En la consola Azure Portal:
1. **Create a resource → Static Web App**
2. Resource group: `priceiq-rg`
3. Name: `priceiq-frontend`
4. Plan: **Free**
5. Region: la misma que el backend
6. Source: **GitHub**
7. Autoriza GitHub si te pide
8. Selecciona tu repo `a00836919/priceiq`, branch `main`
9. Build Presets: **Custom** (no Next.js predefinido)
10. App location: `/frontend`
11. Output location: dejar vacío
12. **Create** → espera 1-2 min

Esto **automáticamente crea un GitHub Action** en tu repo
(`.github/workflows/azure-static-web-apps-XXX.yml`) — bórralo o ignóralo
porque ya tenemos uno mejor (`deploy-frontend.yml`).

### 2.2 Obtener el deployment token

```bash
SWA_TOKEN=$(az staticwebapp secrets list \
    --name priceiq-frontend --resource-group priceiq-rg \
    --query properties.apiKey -o tsv)
echo "$SWA_TOKEN"
```

### 2.3 Configurar GitHub Secrets y Variables

```bash
# Secrets (sensibles)
gh secret set AZURE_STATIC_WEB_APPS_API_TOKEN -b "$SWA_TOKEN"
gh secret set AZURE_SUBSCRIPTION_ID -b "$(az account show --query id -o tsv)"
gh secret set AZURE_TENANT_ID       -b "$(az account show --query tenantId -o tsv)"
```

Para el `AZURE_CLIENT_ID` (autenticación OIDC para el backend):
```bash
# Crear un Service Principal con federación OIDC
SP=$(az ad sp create-for-rbac --name "priceiq-github" \
    --role contributor --scopes "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/priceiq-rg" \
    --json-auth)
echo "$SP" | jq -r .clientId
```

Configura federation OIDC para que GitHub pueda autenticarse sin password (más seguro). Sigue: https://learn.microsoft.com/en-us/azure/developer/github/connect-from-azure-openid-connect

Para empezar rápido, alternativa más simple con password:
```bash
APP_ID=$(echo "$SP" | jq -r .clientId)
gh secret set AZURE_CLIENT_ID -b "$APP_ID"
# y usar el credentials JSON en lugar de OIDC en el workflow
gh secret set AZURE_CREDENTIALS -b "$SP"
```

Variables (no sensibles):
```bash
gh variable set AZURE_RG          -b "priceiq-rg"
gh variable set AZURE_ACR         -b "priceiq12345"   # del output del script
gh variable set AZURE_APP         -b "priceiq-api"
gh variable set NEXT_PUBLIC_API_URL -b "https://TU-URL-DEL-BACKEND"
```

### 2.4 Trigger el primer deploy del frontend

```bash
git add .
git commit -m "Deploy a Azure"
git push origin main
```

Ve a https://github.com/a00836919/priceiq/actions y ve cómo se ejecutan los workflows.

Cuando termine, obtén la URL del frontend:
```bash
az staticwebapp show -n priceiq-frontend -g priceiq-rg \
    --query defaultHostname -o tsv
# → priceiq-frontend.azurestaticapps.net
```

### 2.5 Actualizar CORS en el backend con la URL real del frontend

```bash
az containerapp update --name priceiq-api --resource-group priceiq-rg \
    --set-env-vars "CORS_ORIGINS=https://priceiq-frontend.azurestaticapps.net"
```

---

## 3. Día 4 — Monitoreo y verificación

### 3.1 Application Insights

Ya quedó conectado en el script. Para verlo:

1. Portal Azure → buscar **Application Insights** → `priceiq-appinsights`
2. **Live Metrics** — telemetría en tiempo real (requests/min, latencia, errores)
3. **Failures** — errores con stack trace
4. **Performance** — endpoints más lentos
5. **Logs** — query KQL custom:

```kql
// Top 10 endpoints más usados
requests
| where timestamp > ago(24h)
| summarize count() by name
| top 10 by count_

// Latencia promedio de /api/recommend
requests
| where name == "POST /api/recommend"
| summarize avg(duration), percentile(duration, 95) by bin(timestamp, 1h)
| render timechart
```

### 3.2 Alertas básicas (Portal)

Configura alertas a tu email:

1. **Container App → Alerts → Create**
2. Condition: **Requests Failed (5xx)** > 5 en 15 min
3. Action group: **Email** con tu correo

### 3.3 Costos

```bash
# Ver el gasto actual del Resource Group
az consumption usage list --start-date 2026-05-01 --end-date 2026-05-31 \
    --query "[?contains(instanceName, 'priceiq')]" -o table
```

---

## 4. Operación diaria

### Re-deployar el backend

Solo `git push` a `main` con cambios en `backend/`, `recommender.py` o `Dockerfile`:

```bash
git add backend/main.py
git commit -m "fix: ajustar endpoint X"
git push
# → GitHub Action construye Docker, push a ACR, deploy a Container App. ~3 min.
```

### Re-deployar el frontend

```bash
git add frontend/
git commit -m "feat: nueva página de promociones"
git push
# → Static Web Apps detecta cambio, build Next.js, deploy. ~2 min.
```

### Actualizar datos/modelos (sin re-entrenar)

```bash
# Si tienes data nueva localmente, subirla al File Share
az storage file upload-batch \
    --account-name TUSTORAGE \
    --account-key "$(az storage account keys list -g priceiq-rg -n TUSTORAGE --query '[0].value' -o tsv)" \
    --destination priceiq-artifacts --destination-path "data" \
    --source data --pattern "*.parquet"

# Reiniciar el container para que recargue cache
az containerapp revision restart -n priceiq-api -g priceiq-rg \
    --revision "$(az containerapp show -n priceiq-api -g priceiq-rg --query properties.latestRevisionName -o tsv)"
```

### Ver logs en vivo

```bash
az containerapp logs show -n priceiq-api -g priceiq-rg --follow
```

### Escalar manualmente

```bash
# Mantener siempre 1 réplica viva (evita cold start)
az containerapp update -n priceiq-api -g priceiq-rg --min-replicas 1

# Volver a scale-to-zero (ahorrar dinero cuando no se usa)
az containerapp update -n priceiq-api -g priceiq-rg --min-replicas 0
```

---

## 5. Cómo borrar todo (si quieres ahorrar o empezar de cero)

```bash
# CUIDADO: borra TODOS los recursos del proyecto
az group delete --name priceiq-rg --yes --no-wait
```

Esto **no afecta nada de Static Web Apps** si lo creaste en otro RG. Bórralo
desde el portal o:

```bash
az staticwebapp delete --name priceiq-frontend --resource-group priceiq-rg --yes
```

---

## 6. Checklist final

Antes de presentar/usar productivamente:

- [ ] `curl https://TU-API/api/health` responde `{"status":"ok","ready":true}`
- [ ] Frontend en `https://priceiq-frontend.azurestaticapps.net` carga sin errores
- [ ] Buscar un SKU en frontend → recomendación aparece
- [ ] Subir CSV de ejemplo en `/batch` → tabla de resultados sale
- [ ] Application Insights muestra requests en Live Metrics
- [ ] GitHub Action verde en el último push
- [ ] Una alerta de email configurada para 5xx errors

---

## 7. Solución de problemas comunes

### "Container falla con healthcheck timeout"
- Cold start es 90s+ por la carga de modelos. Aumenta `--start-period` en Dockerfile.
- Verifica logs: `az containerapp logs show -n priceiq-api -g priceiq-rg --tail 100`

### "CORS error en el browser"
- Confirma env var: `az containerapp show -n priceiq-api -g priceiq-rg --query properties.template.containers[0].env`
- Debe incluir la URL exacta del frontend (con https://, sin trailing slash)

### "GitHub Action falla en az login"
- Verifica que `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID` estén en secrets
- Si usaste service principal con password, deberías tener `AZURE_CREDENTIALS` en lugar de los 3 anteriores

### "Modelo no encuentra los archivos"
- Verifica env vars en el container app:
  ```bash
  az containerapp show -n priceiq-api -g priceiq-rg \
      --query properties.template.containers[0].env -o json
  ```
- Debe tener `PRICEIQ_DATA_DIR=/app/data/data` y `PRICEIQ_MODELS_DIR=/app/models/models`
  (sí, doble subcarpeta porque el File Share monta el share completo)

### "ACR build muy lento"
- La primera vez tarda 10-15 min porque pip install descarga lightgbm/catboost/xgboost
- Subsecuentes builds usan cache de Azure → 2-3 min

### "Container se reinicia constantemente"
- Probablemente OOM (out-of-memory). Aumenta memoria:
  ```bash
  az containerapp update -n priceiq-api -g priceiq-rg --memory 4Gi --cpu 2
  ```

---

## 8. Próximos pasos (cuando ya tengas todo arriba)

- **Re-entreno automático**: GitHub Action mensual que corre `build_dataset.py + ... + backtest_quasi.py` y sube nuevos modelos al File Share.
- **Custom domain**: comprar `priceiq.tudominio.com` y apuntarlo al frontend.
- **Auth**: si vas a usar con cliente real, agregar Azure AD B2C al frontend.
- **CDN para frontend**: ya viene built-in con Static Web Apps.
- **Database real**: si necesitas multi-usuario o actualizaciones en vivo, migrar de parquet a Postgres.

---

*Última actualización: mayo 2026 — versión 1.0*
