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
** Este tipo de arquitecturas han demostrado excepcionales resultados capturando complejos patrones en datos de texto
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

## **6.Transformers**
Un transformer es una arquitectura de deep learning para procesar, entender y generar texto en lenguaje humano.
Las características que los hacen especialmente eficaces son:
* Los transformers no usan capas recurrentes como parte de sus componentes de la red neuronal.
* Esto permite capturar dependencias más a largo plazo que las RNNs en textos largos gracias a sus dos características principales
** Mecanismos de atención + Positional Encoding.
** Gracias a estos dos mecanismos los Transformers son capaces de ponderar la importancia relativa de las diferentes palabras en una frase al hacer inferencias, por ejemplo, para predecir la siguiente palabra a generar como una parte de una secuencia de salida.
* Los tokens se procesan simultáneamente:
** Gracias a los mecanismos de atención los Transformers son capaces de manejar los tokens simultáneamente en lugar de secuencialmente, resultando en inferencias y entrenamientos más rápidos.

La arquitectura de un transformer es la siguiente:
![transformer](https://github.com/boresmol/Introduction-to-LLMs-in-Python-Datacamp/blob/main/transformers.png)
