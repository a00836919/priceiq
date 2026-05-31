#!/usr/bin/env bash
# =============================================================================
# 01-azure-setup.sh — Crea toda la infraestructura Azure para PriceIQ.
#
# Ejecutar una sola vez para provisionar:
#   • Resource Group
#   • Storage Account + File Share (para data/ y models/)
#   • Container Registry (para la imagen Docker)
#   • Log Analytics + Application Insights
#   • Container Apps Environment + Container App
#
# Prerequisitos:
#   • az CLI instalado:   brew install azure-cli
#   • az login            # autenticarse en la suscripción
#   • Cambiar las vars de abajo si quieres otro nombre/region
#
# Costo estimado (free tiers): ~$5-10/mes con scale-to-zero
# =============================================================================
set -euo pipefail

# -----------------------------------------------------------------------------
# Variables — personalizar SI quieres otro nombre o región.
# Los nombres globales (storage, ACR) deben ser únicos en Azure → puede que
# necesites agregarle un sufijo si ya existen.
# -----------------------------------------------------------------------------
LOCATION="${LOCATION:-eastus}"
SUFFIX="${SUFFIX:-$RANDOM}"     # números aleatorios para garantizar unicidad
RG="${RG:-priceiq-rg}"
STORAGE="${STORAGE:-priceiq${SUFFIX}}"          # storage account (5-24 alfanum minúsculas)
SHARE="${SHARE:-priceiq-artifacts}"             # file share dentro del storage
ACR="${ACR:-priceiq${SUFFIX}}"                  # container registry (5-50 alfanum)
LOGS="${LOGS:-priceiq-logs}"                    # Log Analytics workspace
APPINS="${APPINS:-priceiq-appinsights}"         # Application Insights
ENV_NAME="${ENV_NAME:-priceiq-env}"             # Container Apps env
APP_NAME="${APP_NAME:-priceiq-api}"             # Container App

echo "═══════════════════════════════════════════════════"
echo "  Provisionando PriceIQ en Azure"
echo "═══════════════════════════════════════════════════"
echo "  Region:           $LOCATION"
echo "  Resource Group:   $RG"
echo "  Storage Account:  $STORAGE"
echo "  Container Reg:    $ACR"
echo "  Container App:    $APP_NAME"
echo "═══════════════════════════════════════════════════"
read -p "¿Continuar? (y/N) " ok
[[ "$ok" =~ ^[Yy]$ ]] || exit 0

# -----------------------------------------------------------------------------
# 1. Resource Group — agrupa todos los recursos para fácil borrado
# -----------------------------------------------------------------------------
echo ""
echo "[1/7] Resource Group…"
az group create --name "$RG" --location "$LOCATION" --output table

# -----------------------------------------------------------------------------
# 2. Storage Account + File Share — para data/ y models/
# -----------------------------------------------------------------------------
echo ""
echo "[2/7] Storage Account + File Share…"
az storage account create \
    --name "$STORAGE" --resource-group "$RG" --location "$LOCATION" \
    --sku Standard_LRS --allow-blob-public-access false --output table

STORAGE_KEY=$(az storage account keys list -g "$RG" -n "$STORAGE" --query "[0].value" -o tsv)
az storage share-rm create \
    --resource-group "$RG" --storage-account "$STORAGE" \
    --name "$SHARE" --quota 5 --output table

# -----------------------------------------------------------------------------
# 3. Subir data y models al File Share
# -----------------------------------------------------------------------------
echo ""
echo "[3/7] Subiendo data/ y models/ al File Share… (puede tardar 1-3 min)"
if [[ -d "data" ]]; then
    az storage file upload-batch \
        --account-name "$STORAGE" --account-key "$STORAGE_KEY" \
        --destination "$SHARE" --destination-path "data" \
        --source "data" --pattern "*.parquet" --output none
    az storage file upload-batch \
        --account-name "$STORAGE" --account-key "$STORAGE_KEY" \
        --destination "$SHARE" --destination-path "data" \
        --source "data" --pattern "*.kv" --output none
    echo "  ✓ data/ subido"
fi
if [[ -d "models" ]]; then
    az storage file upload-batch \
        --account-name "$STORAGE" --account-key "$STORAGE_KEY" \
        --destination "$SHARE" --destination-path "models" \
        --source "models" --output none
    echo "  ✓ models/ subido"
fi

# -----------------------------------------------------------------------------
# 4. Container Registry — para guardar la imagen Docker
# -----------------------------------------------------------------------------
echo ""
echo "[4/7] Container Registry…"
az acr create --name "$ACR" --resource-group "$RG" --sku Basic \
    --admin-enabled true --output table

# -----------------------------------------------------------------------------
# 5. Build & push de la imagen (ACR Tasks la construye en Azure, no en tu Mac)
# -----------------------------------------------------------------------------
echo ""
echo "[5/7] Build & push de la imagen Docker (vía ACR Tasks)…"
az acr build --registry "$ACR" --image "priceiq-backend:v1" \
    --file Dockerfile . --output table

# -----------------------------------------------------------------------------
# 6. Log Analytics + Application Insights — para monitoreo
# -----------------------------------------------------------------------------
echo ""
echo "[6/7] Log Analytics + Application Insights…"
az monitor log-analytics workspace create \
    --resource-group "$RG" --workspace-name "$LOGS" --location "$LOCATION" --output none
LOG_ID=$(az monitor log-analytics workspace show \
    --resource-group "$RG" --workspace-name "$LOGS" --query customerId -o tsv)
LOG_KEY=$(az monitor log-analytics workspace get-shared-keys \
    --resource-group "$RG" --workspace-name "$LOGS" --query primarySharedKey -o tsv)

# Extensión application-insights (instalar si no está)
az extension add --name application-insights --upgrade --only-show-errors 2>/dev/null || true
az monitor app-insights component create \
    --app "$APPINS" --location "$LOCATION" --resource-group "$RG" \
    --workspace "$LOGS" --output none
AI_CONN=$(az monitor app-insights component show \
    -g "$RG" -a "$APPINS" --query connectionString -o tsv)

# -----------------------------------------------------------------------------
# 7. Container Apps Environment + App
# -----------------------------------------------------------------------------
echo ""
echo "[7/7] Container Apps environment + app…"
az extension add --name containerapp --upgrade --only-show-errors 2>/dev/null || true
az containerapp env create \
    --name "$ENV_NAME" --resource-group "$RG" --location "$LOCATION" \
    --logs-workspace-id "$LOG_ID" --logs-workspace-key "$LOG_KEY" \
    --output table

# Adjuntar el File Share al environment (para mount-eo)
az containerapp env storage set \
    --name "$ENV_NAME" --resource-group "$RG" \
    --storage-name "priceiq-storage" \
    --azure-file-account-name "$STORAGE" \
    --azure-file-account-key "$STORAGE_KEY" \
    --azure-file-share-name "$SHARE" \
    --access-mode ReadOnly --output none

ACR_USER=$(az acr credential show -n "$ACR" --query username -o tsv)
ACR_PASS=$(az acr credential show -n "$ACR" --query "passwords[0].value" -o tsv)
ACR_SERVER="${ACR}.azurecr.io"

# Crear el Container App
az containerapp create \
    --name "$APP_NAME" --resource-group "$RG" --environment "$ENV_NAME" \
    --image "${ACR_SERVER}/priceiq-backend:v1" \
    --registry-server "$ACR_SERVER" \
    --registry-username "$ACR_USER" \
    --registry-password "$ACR_PASS" \
    --target-port 8000 --ingress external \
    --min-replicas 0 --max-replicas 2 \
    --cpu 1 --memory 2Gi \
    --env-vars \
        "PRICEIQ_DATA_DIR=/app/data" \
        "PRICEIQ_MODELS_DIR=/app/models" \
        "CORS_ORIGINS=*" \
        "APPLICATIONINSIGHTS_CONNECTION_STRING=$AI_CONN" \
    --output table

# Adjuntar el storage al container (volume mounts)
APP_YAML=$(mktemp)
az containerapp show -n "$APP_NAME" -g "$RG" -o yaml > "$APP_YAML"
# Modificamos el YAML para agregar volumes y volumeMounts
python3 - "$APP_YAML" <<'PYEOF'
import sys, yaml
fp = sys.argv[1]
with open(fp) as f: cfg = yaml.safe_load(f)
template = cfg["properties"]["template"]
container = template["containers"][0]
container["volumeMounts"] = [
    {"volumeName": "data-vol",   "mountPath": "/app/data"},
    {"volumeName": "models-vol", "mountPath": "/app/models"},
]
# Sub-paths apuntando a las subcarpetas del file share
template["volumes"] = [
    {"name": "data-vol",   "storageName": "priceiq-storage", "storageType": "AzureFile", "mountOptions": "dir_mode=0755,file_mode=0644"},
    {"name": "models-vol", "storageName": "priceiq-storage", "storageType": "AzureFile", "mountOptions": "dir_mode=0755,file_mode=0644"},
]
with open(fp, "w") as f: yaml.safe_dump(cfg, f, sort_keys=False)
print("YAML actualizado con volume mounts.")
PYEOF

az containerapp update --name "$APP_NAME" --resource-group "$RG" --yaml "$APP_YAML" --output none

# IMPORTANTE: el mount monta todo el File Share. Como subimos data/ y models/
# como subcarpetas, ajustamos las env vars para apuntar a esas subcarpetas:
az containerapp update --name "$APP_NAME" --resource-group "$RG" \
    --set-env-vars \
        "PRICEIQ_DATA_DIR=/app/data/data" \
        "PRICEIQ_MODELS_DIR=/app/models/models" \
    --output none

# URL pública
APP_URL=$(az containerapp show -n "$APP_NAME" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)

echo ""
echo "═══════════════════════════════════════════════════"
echo "  ✅ DEPLOY COMPLETO"
echo "═══════════════════════════════════════════════════"
echo ""
echo "  Backend API:        https://$APP_URL"
echo "  Healthcheck:        https://$APP_URL/api/health"
echo "  Swagger docs:       https://$APP_URL/docs"
echo ""
echo "  Logs en vivo:"
echo "    az containerapp logs show -n $APP_NAME -g $RG --follow"
echo ""
echo "  Para borrar TODO:"
echo "    az group delete -n $RG --yes --no-wait"
echo ""
echo "  Guarda estos valores para el frontend deploy:"
echo "    NEXT_PUBLIC_API_URL=https://$APP_URL"
echo ""
echo "  Y para los GitHub Actions secrets:"
echo "    AZURE_RG=$RG"
echo "    AZURE_ACR=$ACR"
echo "    AZURE_APP=$APP_NAME"
echo "    AZURE_STORAGE=$STORAGE"
echo "    AZURE_FILE_SHARE=$SHARE"
echo "═══════════════════════════════════════════════════"
