# Introduction to LLMs in Python - Datacamp
## Autor: Borja Esteve 

1.**¿Qué son los LLMs (Large Language Models)?**

Los LLMs son sistemas de inteligencia artificial capaces de entender y generar texto con el fin de completar varias tareas relacionadas con el lenguaje. Algunos LLMs populares son:
* GPT
* BERT
* LLaMA

**2.¿Cuáles son las características claves de estos modelos?**

Este tipo de modelos se han hecho tan populares y gozan de gran eficacia y precisión debido a varios factores:

* Están basados en arquitecturas de Deep Learning, comunmente en *Transformers*
** Este tipo de arquitecturas han demostrado excepcionales resultados capturando complejos patrones en datos de texto
* Los LLMs han conseguido avances significativos en tareas de NLP tales como generación de texto, QA...
* Su naturaleza 'Large': Profundas redes neuronales con una cantidad ingente de parámetros entrenables entrenados en enormes corpus de texto.


**3.Ciclo de desarrollo de los LLMs**
![image.png](https://www.google.com/url?sa=i&url=https%3A%2F%2Fh2o.ai%2Fblog%2F2023%2Fentrenando-tu-propio-llm-sin-programacion%2F&psig=AOvVaw12JAp6BPz7EF0_L5V0Ywqv&ust=1704473268150000&source=images&cd=vfe&opi=89978449&ved=0CBEQjRxqFwoTCLD05vGXxIMDFQAAAAAdAAAAABAX)
Los LLMs tienen un ciclo de desarrollo muy parecido a los sistemas convencionales de Machine learning, difiriendo en la parte de Preentrenamiento y fine-tuning:
* En el preentrenamiento, los modelos aprenden patrones generales de lenguaje y aprenden de un gran y variado dataset. Este proceso es computacionalmente muy costoso y el resultado es un modelo preentrenado de propósito general o también llamado *foundation model*. 
* Este *foundation model* puede ser "afinado" (*fine-tuned*) en un dataset mas pequeño y de un dominio específico, convirtiendo un modelo general en un modelo para casos de uso y aplicaciones específicas.

