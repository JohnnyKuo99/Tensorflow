import tensorflow as tf
from datetime import datetime

t1 = [3.0, 4.0, 5.0]
t2 = [6.0, 6.0, 6.0]


#@tf.function
def computeArea(sides):
    a = sides[:, 0]
    b = sides[:, 1]
    c = sides[:, 2]
    s = (a + b + c) / 2
    areaSquare = s * (s - a) * (s - b) * (s - c)
    return areaSquare ** 0.5


stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
# make a logs directory
logdir = 'logs/demo53/%s' % stamp
writer = tf.summary.create_file_writer(logdir)
tf.summary.trace_on(graph=True, profiler=True)

print(computeArea(tf.constant([t1, t2])).numpy())

with writer.as_default():
    tf.summary.trace_export(name='computeArea', step=0, profiler_outdir=logdir)
    tf.summary.trace_off()