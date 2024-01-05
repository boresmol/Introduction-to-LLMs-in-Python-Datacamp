# Introduction to LLMs in Python - Datacamp
## Autor: Borja Esteve 

### **1.¿Qué son los LLMs (Large Language Models)?**

Los LLMs son sistemas de inteligencia artificial capaces de entender y generar texto con el fin de completar varias tareas relacionadas con el lenguaje. Algunos LLMs populares son:
* GPT
* BERT
* LLaMA

### **2.¿Cuáles son las características claves de estos modelos?**

Este tipo de modelos se han hecho tan populares y gozan de gran eficacia y precisión debido a varios factores:

* Están basados en arquitecturas de Deep Learning, comunmente en *Transformers*
            * Este tipo de arquitecturas han demostrado excepcionales resultados capturando complejos patrones en datos de texto
* Los LLMs han conseguido avances significativos en tareas de NLP tales como generación de texto, QA...
* Su naturaleza 'Large': Profundas redes neuronales con una cantidad ingente de parámetros entrenables entrenados en enormes corpus de texto.


### **3.Ciclo de desarrollo de los LLMs**
Los LLMs tienen un ciclo de desarrollo muy parecido a los sistemas convencionales de Machine learning, difiriendo en la parte de Preentrenamiento y fine-tuning:
* En el preentrenamiento, los modelos aprenden patrones generales de lenguaje y aprenden de un gran y variado dataset. Este proceso es computacionalmente muy costoso y el resultado es un modelo preentrenado de propósito general o también llamado *foundation model*. 
* Este *foundation model* puede ser "afinado" (*fine-tuned*) en un dataset mas pequeño y de un dominio específico, convirtiendo un modelo general en un modelo para casos de uso y aplicaciones específicas.

### **4.Cómo accedes a estos *foundation models***
Podemos acceder fácilmente a modelos LLM desde la librería Hugging Face usando Python. Hugging face provee una librería llamada *transformers* la cual ofrece una API con diferentes niveles de abstracción. El *pipeline transformers* ofrece el mayor nivel de abstracción, por lo que se convierte en la forma más sencilla de usar un LLM.

Solo especificando lo que queremos hacer con el modelo, como por ejemplo, clasificación de sentimientos, el pipeline automáticamente descargará un modelo con sus pesos preentrenados. Esto podemos hacerlo simplemente con esta línea de código:

```python3
from transformers import pipeline
sentiment_classifier = pipeline('text-classification') #aquí especificamos la tarea
```

Después, solo tenemos que pasarle la frase que queremos clasificar:

```python3
output = sentiment_classifier('Hola, soy Borja, estoy muy contento de estar escribiendo esta entrada!')
print(output)
```
Nuestro pipiline mostrará su predicción:

`[{'label':'POSITIVE', 'score': 0.9917237012389}]`

### **5.Tareas los que LLMs pueden realizar**
Podemos dividir las tareas en dos grandes grupos:
* Language Generation: donde se pueden encontrar tareas como generación de texto o generación de código
* NLP: donde se encuentran tareas como clasificación de texto y análisis de sentimiento, traducción, reconocimiento de intención...

Un ejemplo de generación de texto con la librería anterior poddría ser el siguiente:

```python3
llm = pipeline('text-generation')
prompt = 'El barrio Gion de kyoto es famoso por'
output = llm(prompt, max_length = 100)
print(output[0]['generated_text']
```
Para resumir el siguiente texto: *The tower is 324 meters (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side …*, haríamos lo siguiente :
```python3
summarizer = pipeline('summarization', model_name)
outputs = summarizer(long_text, max_length = 50)
print(outputs)
```
Y conseguiríamos el siguiente output:
`[{'summary_text': 'the Eiffel Tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres'}]`

Para un *Question Answering model*:
```python3
qa_model = pipeline("question-answering")
question = "For how long was the Eiffel Tower the tallest man-made structure in the world?"
outputs = qa_model(question = question, context = context)
print(outputs['answer'])
```
A lo que el modelo devolverá:
`41 years`

### **6.Transformers**
Un transformer es una arquitectura de deep learning para procesar, entender y generar texto en lenguaje humano.
Las características que los hacen especialmente eficaces son:
* Los transformers no usan capas recurrentes como parte de sus componentes de la red neuronal.
* Esto permite capturar dependencias más a largo plazo que las RNNs en textos largos gracias a sus dos características principales
** Mecanismos de atención + Positional Encoding.
** Gracias a estos dos mecanismos los Transformers son capaces de ponderar la importancia relativa de las diferentes palabras en una frase al hacer inferencias, por ejemplo, para predecir la siguiente palabra a generar como una parte de una secuencia de salida.
* Los tokens se procesan simultáneamente:
** Gracias a los mecanismos de atención los Transformers son capaces de manejar los tokens simultáneamente en lugar de secuencialmente, resultando en inferencias y entrenamientos más rápidos.

La arquitectura del transformer original, [presentado en este paper](https://arxiv.org/abs/1706.03762), es la siguiente:
![transformer](https://github.com/boresmol/Introduction-to-LLMs-in-Python-Datacamp/blob/main/transformers.png)

* El transformer original cuenta con dos partes principales: el *encoder* y el *decoder*.
* Cada una de estas partes contiene varias capas replicadas de alto nivel, llamadas *Encoder layers* y *Decoder layers*
* Dentro de cada una de estas capas de alto nivel, se aplican **Mecanismos de Atención** seguidos por capas *feed-forward* con el fin de capturar complejos patrones semánticos y dependencias en el texto.

#### El primer Transformer con PyTorch
Para crear un transformer hay que definir varios elementos estructurales:
* `d_model`: La dimensión de los *embeddings* usados en todo el modelo para representar *inputs*, *outputs* e información intermedia.
* `n_heads`: Los mecanismos de atención normalmente tienen varias cabezas que funcionan en paralelo, capturando diferentes tipos de dependencias. De normal se elige un número que sea divisor de la dimensión del modelo.
* `num_encoder_layers,num_decoder_layers`: La profundidad del modelo depende del número de capas del decoder y del encoder.

Un ejemplo simple y no funcional de código podría ser el siguiente:
```python3
import torch
import torch.nn as nn

d_model = 512
n_heads = 8
num_encoder_layers = 6
num_decoder_layers = 6

model = nn.Transformer(d_model = d_model, nhead = n_heads, num_encoder_layers = num_encoder_layers, num_decoder_layers = num_decoder_layers)
```
Las arquitecturas principales de Transformers en la actualidad se dividen en 3 tipos:

| Tipo | Tareas | Modelos |
|------------|------------|------------|
| Encoder-Decoder  | Traducción, resumen de texto...  | T5,BART |
| Encoder-only  | Clasificación de texto, QA... | BERT |
| Decoder-only | Generación de texto, generación de QA | GPT |

### **7.Attention mechanisms and positional encoding**
#### *Attention mechanisms*
Los mecanismos de atención son una pieza clave en el éxito de los Transformers. Anteriores arquitecturas como las RNNs normalmente procesaban secuencias token a token, siendo exitosas capturando los tokens procesados recientemente pero fallando en la captura de relaciones de largo alcance. Los mecanismos de atención consiguen superar esa limitación.
Los Transformers usan una estructura de atención llamada *self attention*, la cual pondera con pesos la importancia de todos los tokens en una secuencia **simultáneamente**. 
Pero hay 'una trampa': los mecanismos de atención requieren información sobre la posición de cada token en la secuencia. 
#### *Positional Encoding*
El *positional encoding* añade información a cada toquen sobre su posición en la secuencia, superando así la limitación comentada anteriormente. Pero, ¿Cómo funciona?
* Dado un token transformado a *embedding* que llamaremos 'E', se crea un vector con valores que describen la posición del token en la secuencia.
** Estos valores se crean únicamente utilizando funciones seno y coseno.
* Después, añadimos este token codificado al *embedding*.
* Una vez hecho esto, el token ya está listo para ser procesado por el mecanismo de atención.

![pos_encoding](https://github.com/boresmol/Introduction-to-LLMs-in-Python-Datacamp/blob/main/pos_enc.png)

Para crear una clase *Positional Encoder* en Python, tenemos los siguientes argumentos:
* `max_seq_length`: máxima longitud de la secuencia
* `d_model`: Dimensión del embedding del transformer
* `pe`: *Positional encoding matrix*
* `position`: indice de la posición en la secuencia
* `div_term`: término para escalar los índices de las posiciones

En cuanto al código, tendríamos algo similar a lo siguiente:
```python3
class PositinalEncoder(nn.Module):
  def __init__(self, d_model,max_seq_length = 512):
      super(PositionalEncoder,self).__init__()
      self.d_model = d_model
      self.max_seq_length = max_seq_length

      pe = torch.zeros(max_seq_length,d_model)
      position = torch.arange(0,max_seq_length, dtype = torch.float).unsqueeze(1)

      div_term = torch.exp(torch.arange((0,d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
      pe[:,0::2] = torch.sin(position * div_term)
      pe[:,1::2] = torch.cos(position * div_term)
      pe = pe.unesqueeze(0) #el unesqueeze se usa para transformar al tamaño del batch
      self.register_buffer('pe',pe)

  def forward(self,x): # este método añade el positional encoding a toda la secuencia del embedding
      x = x + self.pe[:, :x.size(1)]
      return x
```

#### *Anatomía de los mecanismos Self Attention*
Los mecanismos *Self Attention* ayudan a los Transformers a entender las interrelaciones que existen entre las palabras de una secuencia. Gracias a esto, el modelo puede centrarse en las palabras más importantes para una tarea dada. Vamos a explicar como funcionan estos mecanismos:

* Dada una secuencia de *n* tokens proyectados en un embbeding
** Cada embedding es proyectado en 3 matrices de misma dimensión: *Query, Key y Values*.
** Aplicando por separado a cada matriz transformaciones lineales, cada una aprende unos pesos durante el entrenamiento.
** Después, se aplica el producto escalar (*dot product*) o la similaridad de coseno entre cada par de *query-key* en una secuencia para crear una matriz de puntuaciones de atención de cada palabra.
** Una vez calculada la matriz de *attention scores*, aplicamos una softmax con el fin de dar una ponderación a estos *scores*, obteniendo así la *attention weights matrix*.
** Después, esta matriz de pesos se multiplica por la matriz *Values* con el fin de obtener un *token embbeding* actualizado con la información relevante de la secuencia.

![scale_dot_product](https://github.com/boresmol/Introduction-to-LLMs-in-Python-Datacamp/blob/main/scale_dot_product.png)

Este mecanismo de atención solo tiene una cabeza de atención. En la práctica, los Transformers paralelizan múltiples cabezas de atención con el fin de aprender diferentes aspectos semánticos de la oración. Esto se llama *Multi-Headed Attention* y el principio subyacente es similar al de los filtros de las CNNs. Las *Multi-Head Attention* concatenan sus *outputs* y los proyectan linealmente para mantener un *embedding* de dimensiones consistentes.

![multi_head_attention](https://github.com/boresmol/Introduction-to-LLMs-in-Python-Datacamp/blob/main/multi_head_attention.png)

La clase *Multi Head Attention* en PyTorch tiene los siguientes argumentos y funciones:
* `num_heads` número de cabezas de atención, cada uno maneja embeddings de tamaño `head_dims`
* `nn.Linear()`: Establecer transformaciones lineales para entradas de atención y salida.
* `split_heads()`: Parte el input en las diferentes cabezas de atención con los tamaños de tensor correctos
* `compute_attention()`: computa los pesos de atención usando la *softmax*

El código es el siguiente:
```python3
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model,d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0,2,1,3).contiguous().view(batch_size * self.num_heads, -1, self.head_dim)

    def compute_attention(self, query, key, mask = None):
        scores = torch.matmul(query, key.permute(1,2,0))
        if mask is not None:
          scores = scores.masked_fill(mask == 0, float("-1e9"))
        attention_weights = F.softmax(scores, dim=-1)
        return attention_weights

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query = self.split_heads(self.query_linear(query),batch_size)
        key = self.split_heads(self.key_linear(key),batch_size)
        value = self.split_heads(self.value_linear(value),batch_size)

        attention_scores = self.compute_attention(query, key, mask)

        output = torch.matmul(attention_scores, value)
        output = output.view(batch_size, self.num_heads, -1, self.head_dim).permute(0,2,1,3).contiguous().view(batch_size, -1, self.d_model)

        return self.output_linear(output)

```
