#!/bin/bash

TOKEN="bae99fe10135d850f4a9359880696a1cc8207406ee99c6b1aa2449e46f769da34942623c3e85db20df9f5be8475c38a41021bab9df0b4d90873088c7be17ffc5df8fc8c6fad4f9528cf21725c9602693c5f09fca86e9952c7f66f833fb0a6ddea46b4e22472f4811e2210a024bb51f134d19b304a8bd552a79ceba4199db3f16"

BASE_URL="https://cms.petsrs.com.br/api/pets"

DESAPARECIDO="desaparecido"
PROCURASEDONO="procurase_dono"

image_exists() {
    local image_file=$1
    if [ -f "$image_file" ]; then
        return 0
    else
        return 1
    fi
}

download_image() {
    local url=$1
    local image_file=$2
    if ! image_exists "$image_file"; then
        echo "Fazendo download da imagem $image_file..."
        curl -s -o "$image_file" "$url"
    else
        echo "A imagem $image_file já existe. Pulando download."
    fi
}

save_pet() {
    local pet_data=$1
    local pet_id=$(echo "$pet_data" | jq -r '.data.id')
    local origem=$(echo "$pet_data" | jq -r '.data.attributes.origem' | tr ' ' '_')
    local tipo_animal=$(echo "$pet_data" | jq -r '.data.attributes.tipo' | tr '[:upper:]' '[:lower:]' | iconv -f utf-8 -t ascii//translit | tr ' ' '_')
    local image_urls=$(echo "$pet_data" | jq -r '.data.attributes.foto.data[].attributes.url')

    if [ "$origem" == "Desaparecido" ]; then
        folder_path="${DESAPARECIDO}"
    else
        folder_path="${PROCURASEDONO}"
    fi

    folder_path=${folder_path}/${tipo_animal}

    if [ ! -d "$folder_path" ]; then
        mkdir -p "$folder_path"
    fi

    local image_index=1
    for image_url in $image_urls; do
        if [ "$image_index" -eq 1 ]; then
            local image_file="${folder_path}/${pet_id}.png"
        else
            local image_file="${folder_path}/${pet_id}_${image_index}.png"
        fi
        download_image "$image_url" "$image_file"
        echo "Download da imagem ${image_index} do pet $pet_id (Origem: $origem, Tipo: $tipo_animal) concluído."
        image_index=$((image_index + 1))
    done
}

if [ "$#" -ne 2 ]; then
    ID_INICIAL=0
    ID_FINAL=10
else
    ID_INICIAL=$1
    ID_FINAL=$2
fi


echo "Iniciando download das imagens dos pets..."

for (( i=ID_INICIAL; i<=ID_FINAL; i++ )); do
    pet_data=$(curl -s -H "Authorization: Bearer $TOKEN" "${BASE_URL}/${i}?populate=foto")
    if [ $(echo "$pet_data" | jq -r '.data.id') != "null" ]; then
        save_pet "$pet_data"
    fi
done

echo "Download das imagens dos pets concluído."

