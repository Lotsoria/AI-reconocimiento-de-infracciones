# scripts\run_gui.ps1
# Activa la venv de Windows y lanza Streamlit
$venvActivate = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    . $venvActivate
} else {
    Write-Error "No se encontró .venv. Crea el entorno: python -m venv .venv"
    exit 1
}

# Asegura streamlit instalado
pip show streamlit | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Output "Instalando dependencias..."
    pip install -r requirements.txt
}

python -m streamlit run app/gui_streamlit.py