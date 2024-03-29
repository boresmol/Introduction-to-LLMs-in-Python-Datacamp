# Introduction to LLMs in Python - Datacamp
## Autor: Borja Esteve 

### **1.¿Qué son los LLMs (Large Language Models)?**

Los LLMs son sistemas de inteligencia artificial capaces de entender y generar texto con el fin de completar varias tareas relacionadas con el lenguaje. Algunos LLMs populares son:
* GPT
* BERT
* LLaMA

### **2.¿Cuáles son las características claves de estos modelos?**

Este tipo de modelos se han hecho tan populares y gozan de gran eficacia y precisión debido a varios factores:

* Están basados en arquitecturas de Deep Learning, comúnmente en *Transformers*
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
    * Mecanismos de atención + Positional Encoding.
    * Gracias a estos dos mecanismos los Transformers son capaces de ponderar la importancia relativa de las diferentes palabras en una frase al hacer            inferencias, por ejemplo, para predecir la siguiente palabra a generar como una parte de una secuencia de salida.
* Los tokens se procesan simultáneamente:
    * Gracias a los mecanismos de atención los Transformers son capaces de manejar los tokens simultáneamente en lugar de secuencialmente, resultando en          inferencias y entrenamientos más rápidos.

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
    * Estos valores se crean únicamente utilizando funciones seno y coseno.
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
    * Cada embedding es proyectado en 3 matrices de misma dimensión: *Query, Key y Values*.
    * Aplicando por separado a cada matriz transformaciones lineales, cada una aprende unos pesos durante el entrenamiento.
    * Después, se aplica el producto escalar (*dot product*) o la similaridad de coseno entre cada par de *query-key* en una secuencia para crear una             matriz de puntuaciones de atención de cada palabra.
    * Una vez calculada la matriz de *attention scores*, aplicamos una softmax con el fin de dar una ponderación a estos *scores*, obteniendo así la              *attention weights matrix*.
    * Después, esta matriz de pesos se multiplica por la matriz *Values* con el fin de obtener un *token embbeding* actualizado con la información relevante de la secuencia.

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
### **8.Creando un Encoder de un Transformer**
La estructura original de los Transformers combina un encoder y un decoder para la compresión y generación de lenguajes de secuencia a secuencia.
Los Transformers que solo usan encoder simplifican la arquitectura original para escenarios donde el objetivo principal es comprender y representar los datos de entrada en lugar de generar secuencias.
El encoder del Transformer tiene dos componentes principales:
* *Transformer body*: Es un stack de múltiples capas de encoder para aprender patrones complejos en los datos del lenguaje.
* *Encoder layer*: Cada capa del encoder incluye:
    * *Multi-head self-attention* : para capturar las relaciones entre los distintos componentes de las secuencias
    * *Feed Forward Layers*: Para mapear el conocimiento de las capas de atención en representaciones abstractas no lineales.
    * Capas de normalización
    * *Skip connections*
    * Droputs
    * *Transformer head*: Capa final del modelo. Está diseñado para producir resultados específicos de la tarea. En los transformers con solo encoder es                              típico que estas tareas sean clasificar o predecir algo.

  ![encoder](https://github.com/boresmol/Introduction-to-LLMs-in-Python-Datacamp/blob/main/encoder_transformer.png)

  #### Subcapa Feed Forward en la capa del Encoder
Nuestra capa "*FeedForwardSublayer*" contendrá 2 fully-connected + ReLU. Para programar esta clase, usaremos los siguientes parámetros:
* `d_ff`:dimension entre las capas lineales. Es bueno que sea diferente a la dimensión del embedding para facilitar aún más la captura de patrones          
         complejos.
* `forward()`: fuynción que procesa los outputs de atención para capturar patrones complejos y no lineales

```python3
class FeedForwardSubLayer(nn.Module):
   def __init__(self,d_model,d_ff):
      super(FeedForwardSubLayer, self).__init__()
      self.fc1 = nn.Llinear(d_model,d_ff)
      self.fc2 = nn.Llinear(d_model,d_ff)
      self.relu = nn.ReLU()

   def forward(self,x):
      return self.fc2(self.relu(self.fc1(x)))
```
#### Encoder Layer
Finalmente, el encoder layer, como antes se ha comentado, lo compondrán las capas *Feed Forward* junto con las *Multi-headed self-attention*:

```python3
class EncoderLayer(nn.Module):
   def __init__(self,d_model,num_heads,d_ff,dropout):
      super(EncoderLayer,self).__init__()
      self.self_attn = MultiHeadAttention(d_model,num_heads)
      self.feed_forward = FeedForwardSubLayer(d_model,d_ff)
      self.norm1 = nn.LayerNorm(d_model)
      self.norm2 = nn.LayerNorm(d_model)
      self.dropout = nn.Dropout(dropout)

   def forward(self,x,mask): # Este metodo pasa los datos por en encoder layer
      attn_output = self.self_attn(x,x,x,mask)
      x = self.norm1(x + self.dropout(attn_output)
      ff_output = self.feed_forward(x)
      x = self.norm2(x + self.dropout(ff_output))
      return x
```
Como podemos ver, durante el paso hacia delante, se usa una máscara. Esto se hace para evitar el procesamiento de fichas de relleno. A continuación se explica su funcionamiento:

#### *Masking* the attention process
En tareas de NLP donde las secuencias de entrada tienen diferentes longitudes, el *padding* garantiza la igualdad de la longitud en secuencias para un procesamiento por *batches* óptimo, agregando tokens de relleno especiales.
Sin embargo, el mecanismo de atención no debería fijarse en esos tokens de relleno, ya que estos no contienen información relevante para la tarea lingüistica. Es por eso que se usa una máscara de relleno con ceros para las posiciones rellenadas en la secuencia.

#### *Transformer body*
Una vez que hemos definido el *encoder layer*, apilamos varios de estos para definir el cuerpo del Transformer:
```python3 
class TransformerEncoder(nn.Module)
   def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_sequence_length):
      super(TransformerEncoder, self).__init__()
      self.embbeding = nn.Embedding(vocab_size, d_model)
      
      self.positional_encoding = PositionalEncoding(d_model, max_sequence_length)
      self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range (num_layers)])

   def forward(self,x,mask):
      x = self.embedding(x)
      x = self.positional_encoding(x)
      for layer in self.layers:
      x = layer(x,mask)
```

#### *Transformer Head*
Vamos a presentar una cabeza de transformer para tareas de clasificación tales como clasificación de texto, de análisis de sentimiento...
Simplemente consiste en una última capa lineal completamente conectada que mapea los estados ocultos del codificador en probabilidades de clase, con la ayuda de una función softmax.

```python3
class ClassifierHead(nn.Module):
   def __init__(self, d_model, num_classes):
      super(ClassifierHead,self).__init__()
      self.fc = nn.Linear(d_model, num_classes)
   
   def forward(self,x):
      logits = self.fc(x)
      return F.log_soft_max(logits, dim = -1)
```

### **9.Creando un *Decoder Transformer***
La versión *decoder-only* es una versión simplificada de la arquitectura original del Transformer especializada en tareas de generación de secuencias que no requieren *encoder*. 
Concretamente, este tipo de arquitecturas están diseñadas para manejar tareas autorregresivas de generación de secuencias tales como generación y finalización de texto.
La estructura es muy similar a la de un *encoder-only* exceptuando dos diferencias:
* El uso de *Masked multi-head self-attention*: Esto ayuda al modelo a especializarse en predecir la siguiente palabra en una secuencia paso a paso, generar de forma iterativa mensajes, respuestas o cualquier texto como lo hacen los LLMs autorregresivos.
    * *Upper triangular mask*: Para cada token en la secuencia objetivo, solo se observan los tokens generados previamente, mientras que las fichas    
                               posteriores se ocultan mediante el uso de una máscara triangular superior (*Upper triangular mask*) que impide atender a                                    posiciones futuras.
* La otra diferencia radica en el *Transformer Head*: Normalmente en esta arquitectura se trata de una capa con activación softmax sobre todo el       
                                                      vocabulario apra estimar la probabilidad de que cada palabra o token sea el siguiente en generarse y 
                                                      devuelva el más probable.
      
![decoder_only](https://github.com/boresmol/Introduction-to-LLMs-in-Python-Datacamp/blob/main/decoder_only.png)

Vamos a implementar estos dos elementos que difieren de la arquitectura encoder-only.

#### **Masked self-attention**
Esto es clave para aportarle al modelo un comportamiento autorregresivo o causal, y se logra mediante el uso de una máscara de atención triangular.
Al pasar esta matriz a la cabeza de atención, cada token en la secuencia solo presta atención a la información "pasada" en su lado izquierdo.

En el siguiente ejemplo, durante el entrenamiento, el token "favourite" en la secuencia "Orange is my favourite fruit." solo prestaría atención a las fichas anteriores: orange, is, my y favourite. 

![masked_self_attention](https://github.com/boresmol/Introduction-to-LLMs-in-Python-Datacamp/blob/main/mask_self_attention.png)

De este modo, durante la inferencia, el modelo aprenderá que en secuencias como estas, la siguiente palabra a generar podría ser "fruit".
El uso de estas máscaras no requiere modificar el mecanismo de autoatención de múltiples cabezas que se implementó para arquitecturas de tipo *Encoder Only*. En cambio, una vez definida toda la arquitectura, simplemente creamos una máscara triangular que se pasará al modelo junto con la secuencia de entrada:

```python3
self_attention_mask = (1-torch.triu(torch.ones(1,sequence_length, sequence_length), diagonal = 1)).bool()
(...)
output = decoder(input_sequence, self_attention_mask)
```
En cuanto a la clase del Decoder, se puede implementar tanto fuera como dentro de la clase de cuerpo del Transformador. 
```python3
Class DecoderOnlyTransformer(nn.Module):
   def __init__(self,vocab_size, d_model,num_layers,num_heads,d_ff,dropout,max_sequence_length):
      super(TransformerDecoder,self).__init__()
      self.embedding = nn.Embedding(vocab_size, d_model)
      self.positional_encoding = PositionalEncoding(d_model, max_sequence_length)
      self.layers = nn.ModuleList([DecoderLayer( d_model, num_heads,  d_ff, dropout) for _ in range(num_layers)])
      self.fc = nn.Linear(d_model,vocab_size)
   
   def forward (self,x,self_mask):
      x = self.embedding(x)
      x = self.positional_encoding(x)
      for layer in self.layers:
         x= layer(x,self_mask)
      x = self.fc(x)
      return F.log_softmax(x,dim=-1)
```
### ** 10.Construcción de un *encoder-decoder transformer* **

Ya hemos visto como se construyen tanto el *encoder* como el *decoder* pero, ¿Cómo unimos estas dos arquitecturas? La respuesta es otra variante del mecanismo de autoatención, llamado *Cross-atention o Encoder-Decoder attention*, que se agrega en cada capa del decodificador después de la atención enmascarada. Este componente toma dos inputs:
* La información procesada a través del decodificador
* Los estados ocultos producidos por el codificador
Esto es crucial para que el *decoder* 'mire hacia atrás' a la entrada para determinar que generar a continuación en la secuencia objetivo. 

![decoder-cross-attention](https://github.com/boresmol/Introduction-to-LLMs-in-Python-Datacamp/blob/main/decoder-encoder.png)

Ahora el paso forward del decoder necesitará 2 máscaras: la máscara causal para la primera etapa de atención y la máscara de atención cruzada, que puede ser la máscara de relleno habitual como la que se utiliza en el codificador.

Es importante destacar que la variable `y`de este método contiene las salidas del codificador, que se pasan a el mecanismo de atención cruzada como argumentos clave-valor, mientras que el flujo del decodificador asociado a la secuencia objetivo a generar solo adopta el rol de consulta de atención.
```python3
class DecoderLayer(nn.Module):
   def __init__(self, d_model, num_heads, d_ff, dropout):
      super(DecoderLayer, self).__init__()
      self.self_attn = MultiHeadAttention(d_model, num_heads)
      self.cross_attn = MultiHeadAttention(d_model, num_heads)
      ...
   def forward(self, x, y, causal_mask, cross_mask):
      self_attn_output = self.self_attn(x,x,x, causal_mask)
      x = self.norm1(x + self.dropout(self_attn_output)
      cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, cross_mask)
      x = self.norm2(x + self.dropout(cross_attn_output)
      ...
```
En definitiva, la arquitectura *encoder-decoder Transformer* queda así:

![encoder-decoder-transf](https://github.com/boresmol/Introduction-to-LLMs-in-Python-Datacamp/blob/main/encoder-decoder-transf.png)

Un último aspecto importante a considerar es el papel de las entradas del *decoder*, llamadas *output embeddings*.
El *decoder* solo necesita tomar secuencias objetivo reales durante el tiempo de entrenamiento. En tareas de traducción, esto serían ejemplos de traducciones asociados con las secuencias del idioma de origen enviadas al codificador.

Las palabras en la secuencia objetivo actúan como nuestras etiquetas de entrenamiento durante el proceso de generación de la siguiente palabra.

En el momento de la inferencia, el *decoder* asume el papel de generar una secuencia objetivo, comenzando con un *output embedding* vacío y tomando gradualmente como entradas las palabras objetivo que está generando sobre la marcha. 

Vamos a ver un ejemplo de código **simplificado**:
```python3
class PositionalEncoding(nn.Module):
...
class MultiHeadAttention(nn.Module):
...
class FeedForwardSubLayer(nn.Module):
...
class EncoderLayer(nn.Module):
...
class DecoderLayer(nn.Module):
...
```

```python3
class TransformerEncoder(nn.Module):
...
class TransformerDecoder(nn.Module):
...
class ClassificationHead(nn.Module):
...
```
Una vez que todas las clases de componentes estén definidos, podemos creas una clase `Transformer`que encapsule todo:

```python3
class Transformer(nn.Module):
   def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, dropout):
      super(Transformer, self).__init__()
      
      self.encoder = TransformerEncoder(vocab_size, d_model, num_heads, num¨_layers, num_heads, d_ff, max_seq_len, dropout)
      self.decoder = TRansformerDecoder(vocab_size, d_model, num_heads, num_layers, num_heads, d_ff, max_seq_len, dropout)
      
   def forward(self, src, src_mask, causal_mask):
      encoder_output = self.encoder(src, src_mask)
      decoder_output = self.deocder(src, encoder_output, causal_mask, mask)
      
      return decoder_output
```

### ** 11. LLMs para clasificación y generación de texto **
Hemos realizado esto anteriormente con la interfaz `pipeline()`, cuya simplicidad permite utilizar LLMs con muy pocas líneas de código, seleccionando automáticamente un modelo y un tokenizador adecuados.  
Sin embargo, este alto nivel de bastracción permite un menor control y personalización.
La clase `AutoModel` es una alternativa más flexible y personalizable para aporvechar los LLM previamente entrenados. 

Aquí podemos ver un ejemplo de como podemos usar dos clases de `Automodel` comunes en la biblioteca `transformers`de `HuggingFace`
```python3
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

model_name = "bert-based-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

text = 'Soy un ejemplo de secuencia para clasficiación de texto'

class SimpleClassifier(nn.Module):
   def __init__(self, input_size, num_classes):
      super(SimpkleClassifier, self).__input__()
      self.fc = nn.Linear(input_size, num_classes)
   
   def_forward(self, x):
      return self.fc(x)
```
`AutoModel`es una clase genérica que, cuando se le pasan algunas entradas para inferencia, devuelve los estados ocultos producidos por el cuerpo del modelo, pero carece de un encabezado específico para la tarea. Por lo tanto, debemos incluirlo nosotros mismos.

Los pasos serían los siguientes:
* *tokenize* los inputs
* Obtener los estados ocultos del modelos en los `outputs`:
    * `pooler_outputs`: representación agregada de la secuecnia
    * `last_hidden_states`: Estados ocultos desagregados
    *  Forward pass a través de una cabeza de clasificación para obtener la probabilidad de clase.
```python3
inputs = tokenizer(text, return_tensors = 'pt', padding = True, truncation = True, max_length = 64)
output = model(**inputs)
pooled_output = outputs.pooler_output

print('Hidden states size: ' , outputs.last_hidden_state.shape)
print('Pooled output size' , pooled:output.shape)
```
`Hidden states size: torch.Size([1,11,768])`
`Pooled output size: torch.Size([1,768])` 

```python3
classifier_head = SimpleClassifier(pooled_output.size(-1), num_classes = 2)
logits = classifier_head(pooled_output)
probs = torch-softmax(logits, dim = 1)
print(Predicted Class Probabilities:', porbs)
```
`Predicted Class Probabilities: tensor([[0.4334, 0.5666]], grad_fn = <SoftmaxBackward0>)`

Algunos modelos de `AutoModel`aceptan modelos preconfigurados con un cabezal para tareas específicas, lo que elimina la necesidad de agregarlo manualmente:

```python3
from trasnnformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutomodelForSequenceClassification.from_pretrained(model_name)

text = "La calidad del producto era justa"
inputs = tokenizer(text, return_tensor = 'pt')
outputs = model(**inputs)
logits = outputs.logits

predicted_class = torch.argmax(logits, dim=1).item()
print(f'Predicted class index {predicted_cllass + 1} star.")
```
`Predicted class index: 3 star.`

De manera similar, existe una `AutoClass` para generación de texto: 

```python3
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretraiend(model_name)

prompt = ' Esto es un ejemplo simple de generación de texto '
inputs = tokenizer.encode(prompt, return_tensor = 'pt')
output = model.generate(inpiuts, max_legnth = 256)

generated_text = tokenizer.decode(output[0], skip_especial_token = True)
```

### ** 12. ¿Cómo se entrena un LLM para generación de texto?
La predicción de la siguiente palabra es una forma de tarea autosupervisada que requieren ejemplos de entrenamiento que consisten en pares de secuencias de *input-label*. 

Un *input* es una secuencia que representa una parte de un texto. 
La secuencia objetivo o *label* es la misma que la de entrada pero desplazada en un token a la izquierda para que aparezca el token inmediatamente siguiente en la secuencia original

Por ejemplo:

![ejemplo-dato](https://github.com/boresmol/Introduction-to-LLMs-in-Python-Datacamp/blob/main/ejemplo_dato.png)

### ** 13. LLMs para resumen y traducción de texto **
En este capítulo se va a explorar el resumen y la traducción de texto mediante LLMs.

##### Resumen de texto

En el caso de ** resumen de texto ** el obejtivo es generar un resumen de un texto de entrada, preservando información importante sobre su significado. Entrenar un modelo para esta tarea requiere pares de *input-target* donde tanto la entrada como el target son secuencias que contienen un texto original y su texto resumido asociado. Hay dos tipos de procesos de resumen:
* Resumen extractivo: selecciona, extrae y combina partes del texto original para crear un resumen, utilizando modelos de codificador, como BERT.
* Resumen abstractivo: se basa en LLM de secuencia a secuencia (seq2seq son modelos usados para abordar tareas en las que la longitud de entrada y salida                          no son iguales) para generar (palabra por palabra) un resumen que puede utilizar palabras y estructuras de oraciones diferentes a                           las del texto original.

```python3
from transformer import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = 't5-small'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelSeq2SeqLM.from_pretrained(model_name)

input_ids = tokenizer.encode('summarize: ' + example['Article'], return_tensor = 'pt', max_length = 512, truncation = True)

summary_ids = model.generate(input_ids, max_length = 150)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens = True)
```

##### Language Translation
El objetivo de esta tarea es producir una versión traducida de un texto, manteniendo el mismo significado y preservando el contexto.
Los datos que se utilizan son pares *input-target* donde el input es el texto en el idioma original y el target es el texto en el idioma destino. 
La traducción normalmente es posible ghracias a modelos *encoder-decoder* como el Transformer original. 
La secuencia original (*inpiut*) está codificada en una representación numérica, que el decodificador mapeará a una traducción del lenguaje destino.

![traduccion](https://github.com/boresmol/Introduction-to-LLMs-in-Python-Datacamp/blob/main/traduccion.png)

```python3
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = 'Helsinki-NLP/opus-mt-en-cy'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

input_seq = "2 regulations under section 1: supplementary"
input_ids = tokenizer.encode(input_seq, return_tensors='pt')
translated_ids = model-generate(input_ids)
translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens = True)
```

### ** 14. LLMs para *Question-Answering (QA)* **
Las tareas de QA o control de calidad pueden adoptar varias formas:
* *Extractive QA*: El LLM recibe una pregunta junto a un contexto y la respuesta debe extraerse directamente del contexto. Esta es una tarea de aprendizaje supervisada que requiere de una arquitectura ***encoder only***
* *Open Generative QA*: Recibe un contexto pero aplica la generación del lenguaje para construir la respuesta con la ayuda del contexto, en lugar de extraer la respuesta como parte del contexto. Se basa en arquitecturas ***encoder-decoder***
* *Closed generative QA*: Implica una arquitectura ***decoder only*** genera la respuesta sobre el conocimiento del modeelo, sin ningún contexto.

##### Extractive QA
Esta tarea se puede ver como una tarea supervisada de clasificación. La pregunta preprocesada y el contexto se aprueban conjuntamente como entrada al LLM que devuelve algunas salidas sin procesar o logits. Hay dos logits de salida generados para cada token de entrada en la secuencia de entrada, indicando la probabilidad de que el token constituya la posición inicial o final del intervalo de respuestas. Los logits sin procesar se procesan posteriormente para obtener la predicción real o el intervallo de respuesta: una parte de la secuencia de entrada definida por las posiciones de token inicial y final que probablemente contengan la respuesta. Este intervalo de respuestas se obtiene como las posiciones de los logits incial y final con la mayor probabilidad combinada. Veamos el código:

```python3
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_ckp = 'deepset/minilm-uncased-squad2'
tokenizer = AutoTokenizer.from_pretrained(model_ckp)

question = 'Cual es el sabor del wasabi?'
context = 'La cocina japonesa captura la esencia de una armoniosa fusion entre los ingredientes frescos y las técnicas de cocina tradicional, todo proporcionado por el aromático sabor del wasabi'

inputs = tokenizer(question,context, return_tensors = 'pt')

model = AutoModelForQuestionAnswering.from_pretrained(model_ckp)

with torch.no_grad():
   outputs = model(**inputs)

stat_idx = torch-argmax(outputs.start_logits)
end_idx = torch.argmax(outputs.end_logits) + 1

answer_span = inputs['inpiut_ids'][0][start_idx:end_idx]
answer = tokenizer.decode(answer_span)
```

### ** 15. LLM fine-tuning y Transfer Learning **
En los anteriores capítulos hemos utilizado LLMs entrenados para una variedad de propósitos. Ahora vamos a ver como pueden ajustarse estos modelos para un caso de uso específico.
El *fine tuning* implica tomar un modelo previamente entrenado y volver a entrenarlo para resolver una tarea posterior particular con datos de dominio específico. El objetivo es mejorar su desempeño para esa tarea.
Por ejemplo, tomemos un modelo de resumen de propósito general que se ajusta a un conjunto de datos de artículos científicos de química y sus resúmenes para especializarse en resumir artículos de química.
Hay dos enfoques de *fine tuning* diferentes dependiendo de como se actualizan los pesos del modelo:
* *Full fine tuning*: Implica actualizar los pesos de todo el modelo.
* *Partial fine tuning*: Los cuerpos en las capas inferiores del cuerpo del modelo que son responsables de la captura de la comprensión general del lenguaje permanece fija y se actualizan las capas específicas de la tarea solo en el encabezado del modelo.

La elección del enfoque depende de su caso de uso específico, la cantidad de datos específicos de la tarea que se posee y las capacidades informáticas del Hardware.

Un concepto estrechamente ligado al *fine tuning* es el ***Transfer Learning***. El *Transfer Learning* consiste en tomar un modelo entrenado previamente en una tarea y adaptarlo para una tarea diferente pero relacionada. En el contexto de los LLM, esto significa ajustar un conjunto de datos más pequeño pa<ra una tarea particular, como se vio anteriormente.

¿Son el *fine tuning* y el *Transfer Learning* lo mismo?

No exactamente. *Transfer Learning* es un paradigma general sobre el aprovechamiento del conocimiento obtenido en un dominio para mejorar el rendimiento en otro dominio relacionado. Esto puede hacerse mediante *fine tuning* pero también se puede hacer con otros enfoques:

![transf_learn](https://github.com/boresmol/Introduction-to-LLMs-in-Python-Datacamp/blob/main/transfer_learning.png)

* *Zero Shot Learning*: Es un enfoque popular cuando se disponen de pocos datos etiquetados. Un modelo está entrenado para generalizar a nuevas tareas nunca vistas durante el entrenamiento.
* *One shot, few - shot learning*: Se expone el modelo a uno o pocos ejemplos específicos.

#### Fine-Tuning un modelo preentrenado
```python3
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 2)

def tokenize_function(examples):
   return tokenizer(examples["text"], padding = "max_length", truncation = True)

data = load_dataset('imdb')
tokenized_data = data.map(tokenize_function, batched = True)

from transformer import Trainer, TrainingArguments

training_args = TrainingArguments(
      output_dir = "./smaller_bert_finetuned",
      per_device_train_batch_size = 8,
      num_train_epoch = 3,
      evaluation_strategy="steps",
      eval_steps = 500,
      save_steps = 500,
      logging_dir = "./logs")

trainer = Trainer(
      model = model,
      args = training_args,
      train_dataset = tokenized_datasets['train],
      eval_dataset = tokenized_datasets['test']
)

trainer.train()

model.save_pretrained('./my_bert_finetuned')
tokenizer.save_pretrained('./my_tokenizer_finetuned')
```

### **16.Métricas para la evaluación de LLMs**
En esta sección se explorarán métricas y pautas para la evaluación de LLMs. La biblioteca `Evaluate` de `Hugging Face``proporciona una serie de métricas muy útiles para evaluar el rendimiento de este tipo de modelos. La librería contiene 3 métodos:
* `Metric`: una colección de métricas para evaluar el rendimiento del modelo.
* `Comparison` : un conjunto de herramientas para comprar y medir diferencias entre modelos.
* `Measurement`: Centrado en evaluar y obtener información a partir de conjuntos de datos lingüisticos

A partir de ahora nos centraremos principalmente en el paquete `Metric`.
* el atributo `.features`de una métrica nos da información de las entradas necesarias para su cálculo
* `.load("metrica")`se usa para cargar una métrica concreta
* `.compute(etiquetas_reales, predicciones)`se usa para computar la métrica

Las métricas que se usan en cada tipo de tarea son:

![metr](https://github.com/boresmol/Introduction-to-LLMs-in-Python-Datacamp/blob/main/metricas.png)

Algunas de las métricas para tareas especiales de LLM son:
* *Perplejidad*: es una métrica usada para evaluar LLM autorregresivos, como los de generación de texto. Mide la capacidad del modelo para predecir la                       siguiente palabra con precisión y confianza. El rango va de `[0, inf)` y cuando más bajo mejor. Se calcula mediante las distribuciones                      logit de salida devueltas por el modelo. Cuando se pasan múltiples predicciones de texto generadas, es frecuente evaluar la perplejidad                     promedio de las mismas. El código python es el siguiente:
    * ```python3
      perplexity = evaluate.load('perplexity', module_type = 'metric')
      results = perplexity.compute(model_id='gpt2', predictions = generated_text)
      ```
* *ROUGE score* en resúmenes de texto: Cuantifica la superposición y similitud entre un texto resumido generado por el modelo y resúmenes de referencia proporcionados. Presta atención a aspectos como la co-ocurrencia de n-gramas o grupos de tokens consecutivos y la superposición de palabras. Rouge toma una colección de resúmenes previstos o resultados de LLM, como así uno o varios resúmenes de referencia proporcionados por humanos. ROUGE proporciona un conjunto de puntuaciones métricas que capturan diferentes aspectos de similitud del texto, como superposición de unigramas y bigramas, subsecuencias comunes más largas...
* *BLEU score* para traducción: mide la calidad de la traducción entre la salida del LLM y referencias humanas. El valor va entre `0-1`, indicando la similaridad a la referencia (cuanto más cercano a uno, mejor).
* *METEOR score* para traducción: supera algunas limitaciones de *BLEU* y *ROUGE* incorporando más aspectos lingüisticos. Con estas variaciones se consigue lidiar con variaciones morfológicas, capturar palabras con significados similares y penalizar errores en el orden de las palabras. El valor va entre `0-1`y cuanto más alto mejor.
* *Exact Match* en QA: es una métrica sensible. Vale `1` si el output del LLM es exactamente igual a la respuesta referencia y `0`en cualquier otro caso. Suele usarse en conjunto con el `F! score`.


### **17.*Fine-Tuning* del modelo usando *feedback* humano**
La retroalimentación humana puede resultar crucial en el contexto de los LLM por varias razones. Dado que las métricas no pueden capturar de forma correcta la calidad de la subjetividad o el contexto de un LLM, necesitamos retroalimentación humana, usando esta como función de pérdida. Así surge el **RLHF** o *Reinforcement Learning from Human Feedback*. Esto se basa en la idea de que un agente aprenda a tomar decisiones en base a 'premios', adoptando un comportamiento que le permita maximizar la cantidad de 'premios' a lo largo del tiempo. El RLHF consta de tres elementos:
* Un *LLM* inicial: este LLM habrá sido entrenado previamente o incluso puede habérsele hecho *fine tuning* en un conjunto de datos específicos al problema.
* Un modelo de recompensa (*Reward Model*): Un modelo de recompensa entrenado por separado según las preferencias humanas para aprender a asignar recompensas a los resultados LLM.
* Un algoritmo de aprendizaje por refuerzo (por ejemplo, optimización de políticas próximas): Este se usa para guiar el proceso de optimización de LLM actualizando los pesos del modelo a maximizar las recompensas acumuladas esperadas, en función de la política aprendida por el modelo de recompensa.

Este proceso suele ser iterativo: el LLM optimizado se utiliza para producir nuevos datos de texto, que luego se utilizan para volver a entrenar el modelo de recompensa y, posteriormente, seguir ajustando el LLM nuevamente mediante el aprendizaje por refuerzo.

![rlhf](https://github.com/boresmol/Introduction-to-LLMs-in-Python-Datacamp/blob/main/RLHF.png)

Vamos a examinar como se construye un *Modelo de recompensa*:

1. Tener un modelo LLM preentrenado que genere texto.
    * Coleccionar muestras de los inputs-outputs del LLM
2. Basado en procesos de anotación humana para calificar y clasificar estas muestras de LLM, se crea un conjunto de datos para entrenar un modelo de recompensa. Las instancias de capacitación consisten en pares de muestra-recompensa, incorporando así información de preferencias humanas en el proceso.
3. Entrenar un Modelo de Recompensa, capaz de predecir la recompensa de un input-output de un LLM.
4. Una vez entrenado, dada una secuencia de texto, el modelo de recompensa generará una predicción de recompensa escalar. Todo el ciclo se cierra mediante un algoritmo de aprendizaje por refuerzo, que se usa para optimizar el modelo de lenguaje original con respecto al modelo de recompensa.
 
![reward_model](https://github.com/boresmol/Introduction-to-LLMs-in-Python-Datacamp/blob/main/reward_model.png)

#### TRL: Transformer Reinforcement Learning
Es una biblioteca de aprendizaje que se adapta a varios enfoques de RL para ajustar los LLM basados en Transformers. Vamos a enseñar como se añade un modelo de RL basado en PPO (*Proximal Policy Optimization*) que recibe del LLM un triplete: promt, respeusta y recompensa:

```python3
from trl import PPOTrainer, PPOConfig, create_reference_model, AutoModelForCausalLMWithValueHead
from trl.core import respond_to_batch

model = AutoModelForCausalLMWithValueHead.from_pretrained('gpt2')
model_ref = create_reference_model(model)
tokenizer = AutoTokenizer.from_pretrained('gpt2')
if tokenizer.pad_token is None:
   tokenizer.add_special_tokens({'pad_token':'[PAD]'})

prompt = 'My plan today is to'
input = tokenizer.encode(query_txt, return_tensors'pt')
response = respond_to_batch(model,input)

ppo_config = PPOConfig(batch_size = 1)
ppo_trainer = PPOTrainer(ppo_config, model, model_ref, tokenizer)
reward = [torch.tensor(1.0)]
train_stats = ppo_trainer.step([input[0]], [response[0]], reward)
```

### **18.Retos y consideraciones éticas**
En este punto se analizan algunos desafíos y consideraciones éticas asociadas con los LLM.

##### Desafíos de los LLM en el mundo real
Debido a su sotisficación y su potencia, los LLM no solo plantean desafíos comunes a cualquier sistema de IA, sino que también enfrentan algunos desafíos únicos:
1. Uno está ligado a la accesibilidad global a las soluciones LLM, concretamente al permitir el soporte en varios idiomas. Esto implica abordar la diversidad lingüistica, la disponibilidad de recursos y garantizar una tranferibilidad efectiva entre idiomas con diferentes características.
2. El dilema entre los LLM abiertos y cerrados implica lograr el equilibrio entre los beneficios del código abierto, la colaboración y la transparencia de los modelos.
3. La ampliación de los LLM exige mayores capacidades de representación lingüistica, demanda computacional y requisitos de capacitación en términos de costo y acceso a los datos.
4. Los sesgos son otros desafío importante, especialmente si los datos de entrenamiento están sesgados, lo que provoca modelos con mecanismos de generación del lenguaje injusto o discriminatorios.

Además, entre los desafíos más grandes también encontramos:
* Alucinaciones: se dice que un LLM alucina cuando genera información falsa o sin sentido, afirmando que es veraz o exacta.
Las alucinaciones son difíciles de eliminar, pero existen estrategias prácticas para reducirlas:
1. Garantizar que el modelo se base en datos diversos, incluidas varias perspectivas sobre el mismo tema.
2. Auditorías de sesgo y técnicas de mitigación, como el remuestreo.
3. *Fine tune* en casos de uso específico.
4. *Prompt engineering*: El proceso de elaborar y perfeccionar indicaciones adecuadas para obtener respuestas del modelo precisas.


