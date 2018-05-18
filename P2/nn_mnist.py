import gzip
import pickle as cPickle

import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f, encoding='latin1')
f.close()

train_x, train_y = train_set

valid_x, valid_y = valid_set

test_x, test_y = test_set

# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.pyplot as plt

# TODO: the neural net!!

# Las etiquetas están en la última fila. Las codificamos con el one_hot
train_y = one_hot(train_y, 10)


# Placeholder, esto es un tipo de variable que estará constantemente cambiando
# Es decir que en un inicio esta vacía, pero a medida que avanzamos la vamos modificando
# Normalmente se usan para los inputs
# Matriz de entrada
x = tf.placeholder("float", [None, 784])    # samples = muestra [El 784 viene de la multiplicacion de los pixeles de las imagenes 28*28]

# labels = etiqueta [Se pone (9) ya que representa el numero que es cada muestra en forma de vector(one_hot)]
# Matriz con las etiquetas REALES del set de datos mnist
y_ = tf.placeholder(tf.float32, [None, 10])

# ESTA ES LA CAPA OCULTA (INPUT)
W1 = tf.Variable(np.float32(np.random.rand(784, 10)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

# ESTA ES LA SALIDA DE LAS NEURONAS (OUTPUT)
W2 = tf.Variable(np.float32(np.random.rand(10, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)



h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)    # Capa de salida
y = tf.nn.sigmoid(tf.matmul(h, W2) + b2)    # Salida

# -------------------------------------------------------------------------------------
# Aquí estamos pidiendo que se reduzca el error
loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01
# -------------------------------------------------------------------------------------


# Este array se encarga de decirlo que numeros clasifico bien y cuales mal
# Basicamente compara el resultado obtenido contra el resultado teorico, si esta bien GG si no GET REKT
prediccion = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# Nos devuelve un porcentaje(reduce_mean) de certeza
accuracy = tf.reduce_mean(tf.cast(prediccion, "float"))



# Preguntar por el warning y cambie el tf.initialize_all_variables() -> tf.global_variables_initializer()
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)


print ("----------------------")
print ("   Start training...  ")
print ("----------------------")

batch_size = 20

lista_certeza = []
error_validacion = []
lista_error_entrenamiento = []
tasa_error_entrenamiento = 0
error_anterior = 0
estabilidad = 0
epoch = 0

while epoch < 100:
    for jj in range(int(len(train_x) / batch_size)):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
        tasa_error_entrenamiento = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys}) / batch_size



    # Porcentaje de acierto
    tasa_certeza = sess.run(accuracy, feed_dict={x: valid_x, y_: one_hot(valid_y, 10)})

    # Error sobre la validacion
    error_de_validacion = sess.run(loss, feed_dict={x: valid_x, y_: one_hot(valid_y, 10)})/len(test_x)


    if(abs(error_de_validacion - error_anterior) < 0.1):
        estabilidad+=1
        if(estabilidad == 30):break
    else:
        estabilidad = 0
        error_anterior = error_de_validacion

    print("\nEstabilidad: ", estabilidad)


    # Solo quiero 2 decimales en la certeza (que tan bien se hizo el reconocimiento)
    print("\n*- Epoca: {} \n*- Porcentaje de acierto: {:.4}%\n*- Error sobre el set de validacion: {:.4}"
          "\n*- Error del entrenamiento: {:.4}".format(epoch, tasa_certeza*100, error_de_validacion,tasa_error_entrenamiento))

    lista_certeza.append(tasa_certeza)
    error_validacion.append(error_de_validacion)
    lista_error_entrenamiento.append(tasa_error_entrenamiento)

    result = sess.run(y, feed_dict={x: batch_xs})
    #print("Certeza: \n",lista_certeza,"\nValidacion: \n",error_validacion,"\nEntrenamiento: \n",lista_error_entrenamiento)
    for b, r in zip(batch_ys, result):
        print(b, "-->", r)
    print("----------------------------------------------------------------------------------")
    epoch = epoch + 1

# Para añadir otra grafica encima de otra basta con poner en la misma funcion plot, otra lista al final
plt.plot(error_validacion)
plt.plot(lista_error_entrenamiento)
plt.plot(lista_certeza)
plt.legend(["Error validacion","Error entrenamiento","Porcentaje de acierto"]);
plt.show()