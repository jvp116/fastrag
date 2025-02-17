# fastrag

## Descrição

O projeto `fastrag` é uma aplicação que extrai texto de documentos PDF e gera respostas baseadas nesses documentos. Utiliza um modelo local do Ollama para fornecer respostas.

## Funcionalidades

- Extração de texto de arquivos PDF.
- Divisão do texto extraído em blocos para melhor contexto.
- Geração de respostas a perguntas dos usuários com base nos argumentos extraídos.
- Integração com ChromaDB para armazenamento e recuperação de embeddings de texto.

## Pré-requisitos

- Python 3.x
- Bibliotecas necessárias:
  - `fitz` (PyMuPDF)
  - `requests`
  - `chromadb`
  - `fastapi`
  - `pydantic`
  - `sentence-transformers`
- Instalação do Ollama com o modelo apropriado.

## Instalação

1. Clone o repositório:
   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd fastrag

### Utilização
Para rodar a api: uvicorn fastrag:app --reload
curl -X POST "http://localhost:8000/query" -H "Content-Type: application/json" -d '{"query": "Sua pergunta aqui"}'
