# Training code

########### Theano version ###############

# Set things up the way I like it
%load_ext autoreload
%autoreload 2
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

from numpy import *
import time

import theano
import lasagne

# Define epsilon
eps = finfo( float32).eps;

# Optimization toolbox
import downhill

# Some graphics
import pylab
from matplotlib.pyplot import *
from IPython.display import clear_output, display

def drawnow():
    clear_output( wait=True), show()

def imagesc( x, cm = None):
    if cm is None : 
        if amin( x) >= 0 or amax( x) <= 0 : 
            imshow( x, aspect='auto', origin='lower', interpolation='nearest', cmap='bone_r')
        else :
            imshow( x, aspect='auto', origin='lower', interpolation='nearest', cmap='RedBlue')
            clim( amax( fabs(x)), -amax( fabs(x)) )
    else : 
        imshow( x, aspect='auto', origin='lower', interpolation='nearest', cmap=cm)
#     colorbar()
    
# Plot to make while training
def pl():
    clf()
    gcf().set_size_inches(6,2)
    semilogy( cst); grid( 'on'); title( 'Cost: %f, Epoch: %d' % (cst[-1], len( cst)))
    drawnow()

# Read wavfile
import scipy.io.wavfile as wavfile
from sounddevice import *

# Read from deep_sep_expr3.py
from deep_sep_expr3 import sound_feats

# Activation to use
def act( x):
    return theano.tensor.nnet.relu( x)

def psoftplus( x, p = 1.):
    return Th.switch( x < -30./p, 0., Th.switch( x > 30./p, x, Th.log1p( Th.exp( p*x))/p))


############ Load input audio ###############
# Load piano audio
from scipy.io import wavfile
sr,s1 = wavfile.read('/Users/svnktrm2/Dropbox/NN_NMF/Audio/Mary.wav');

sz = 1024; # Window size
hp = sz/4; # Hop size
wn = wn = reshape( hanning(sz+1)[:-1], (sz,1))**.5
FE = sound_feats( sz, hp, wn);

x, Px = FE.fe( s1 / std( s1));

# Plot Spectrogram
imagesc(x**0.4)

# Define NAE network

# Latent dimensions
Kx = 3;

# I/O container
X = theano.tensor.matrix('X')

# Weight matrices
W1x = theano.shared( random.rand( Kx, x.shape[0]).astype( float64))
W2x = theano.shared( random.rand( x.shape[0], Kx).astype( float64))

# Get latent variables
Hx = psoftplus( W1x.dot( X), 2)
# Hx = act( W1x.dot( X))

# Get approximation
Zx = psoftplus( W2x.dot( Hx), 2)
# Zx = act( W2x.dot( Hx))

# Low rank reconstruction should match smoothed amplitudes, use sparse W1
cost = theano.tensor.mean( X * (theano.tensor.log( X+eps) - theano.tensor.log( Zx+eps)) - X + Zx) \
       + 0.01*theano.tensor.mean( abs( Hx)**1) + 1*theano.tensor.mean( abs( W2x)**2) 

# Make an optimizer and define the inputs
opt = downhill.build( 'rprop', loss=cost, params = [W1x, W2x], inputs=[X])
train = downhill.Dataset( x.astype( float64), batch_size = x.shape[0])


# Train and show me the progress
ep = 3000;
cst = []
lt = time.time()
for tm, _ in opt.iterate( train, learning_rate=.001, max_updates=ep):
    cst.append( tm['loss'])
    if time.time() - lt > 1:
        pl()
        lt = time.time()
pl()


# Show me

nn_nmf = theano.function( inputs=[X], outputs=[Zx,Hx], updates = [])
z,h = nn_nmf( x.astype( float64))

subplot( 2, 2, 2); imagesc( x**.4); title( 'Input'); xlabel('Time'); ylabel('Frequency');
basis = W2x.get_value().T

subplot( 2, 2, 1); title( 'NN bases'); xlabel('Component number'); ylabel('Frequency'); 
plot(ones(basis[0,:].shape)-(basis[0,:]/max(basis[0,:])),arange(0,basis[0,:].shape[0])); 
plot(2.5*ones(basis[1,:].shape)-(basis[1,:]/max(basis[1,:])),arange(0,basis[1,:].shape[0])); 
plot(4*ones(basis[2,:].shape)-(basis[2,:]/max(basis[2,:])),arange(0,basis[2,:].shape[0])); 
pylab.ylim([0,512]); pylab.xlim([0,5]); xticks([1,2.5,4],[1,2,3])

subplot( 2, 2, 4); title( 'NN activations'); xlabel('Time'); ylabel('Component number'); 
plot(arange(0,h[0,:].shape[0]),ones(h[0,:].shape)+(h[0,:]/max(h[0,:]))); 
plot(arange(0,h[1,:].shape[0]),2.5*ones(h[1,:].shape)+(h[1,:]/max(h[1,:]))); 
plot(arange(0,h[2,:].shape[0]),4*ones(h[2,:].shape)+(h[2,:]/max(h[2,:]))); 
pylab.xlim([0,h.shape[1]]); pylab.ylim([0,5]); yticks([1,2.5,4],[1,2,3])

subplot( 2, 2, 3); imagesc( z**.4); title( 'Reconstruction'); xlabel('Time'); ylabel('Frequency')

tight_layout()



# Testing code
import time
from numpy import *
from IPython.display import clear_output, display
from matplotlib.pyplot import *

from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, ElemwiseSumLayer
from lasagne.layers import SliceLayer, get_output, get_all_params, Conv1DLayer
from lasagne.layers import RecurrentLayer, get_output_shape
from lasagne.regularization import l1, regularize_layer_params

# Get a Lasagne layer output
def nget( x, s, y):
    if type( s) is list:
        return theano.function( s, squeeze( get_output( x, deterministic=True)), on_unused_input='ignore')( y)
    else:
        return theano.function( [s], squeeze( get_output( x, deterministic=True)), on_unused_input='ignore')( y)
# Separate mixture given NN models
def nn_sep( M, W1, W2, hh = .0001, ep = 5000, d = 0, sp =.0001, spb = 3, al='rprop'):

    # Sort out the activation
    from inspect import isfunction
    if isfunction( spb):
        act = spb
    else:
        act = lambda x: psoftplus( x, spb)

    # Get dictionary shapes
    K = [W1.shape[0],W2.shape[0]]

    # GPU cached data
    _M = theano.shared( M.T.astype( float64))
    dum = Th.vector( 'dum')

    # We have weights to discover
    H = theano.shared( sqrt( 2./(K[0]+K[1]+M.shape[1]))*random.rand( M.T.shape[0],K[0]+K[1]).astype( float64))
    fI = InputLayer( shape=(M.T.shape[0],K[0]+K[1]), input_var=H)

    # Split in two pathways
    fW1 = SliceLayer( fI, indices=slice(0,K[0]), axis=1)
    fW2 = SliceLayer( fI, indices=slice(K[0],K[0]+K[1]), axis=1)

    # Dropout?
    dfW1 = DropoutLayer( fW1, dum[0])
    dfW2 = DropoutLayer( fW2, dum[0])

    # Compute source modulators using previously learned dictionaries
    R1  = DenseLayer( dfW1, num_units=M.T.shape[1], W=W1.astype( float64),
      nonlinearity=act, b=None)
    R2  = DenseLayer( dfW2, num_units=M.T.shape[1], W=W2.astype( float64),
      nonlinearity=act, b=None)

    # Add the two approximations
    R = ElemwiseSumLayer( [R1, R2])

    # Cost function
    Ro = get_output( R)+eps
    cost = (_M*(Th.log(_M+eps) - Th.log( Ro+eps)) - _M + Ro).mean() \
       + sp*Th.mean( abs( H)) + 0*Th.mean( dum)

    # Train it using Lasagne
    opt = downhill.build( al, loss=cost, inputs=[dum], params=[H])
    #train = downhill.Dataset( array( [0]).astype(float32), batch_size=0)
    if isinstance( d, list):
        train = downhill.Dataset( array([d[0]]).astype(float64), batch_size=0)
        er = downhill_train( opt, train, hh, ep/2, None)
        train = downhill.Dataset( array([d[1]]).astype(float64), batch_size=0)
        er += downhill_train( opt, train, hh, ep/2, None)
    else:
        train = downhill.Dataset( array([d]).astype(float64), batch_size=0)
        er = downhill_train( opt, train, hh, ep, None)

    # Get outputs
    _r  = nget( R,  dum, array( [0]).astype(float64)).T + eps
    _r1 = nget( R1, dum, array( [0]).astype(float64)).T
    _r2 = nget( R2, dum, array( [0]).astype(float64)).T

    return _r,_r1,_r2,er
