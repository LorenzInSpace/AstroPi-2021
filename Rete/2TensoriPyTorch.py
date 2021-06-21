import torch

#stessa regressione lineare del codice precedente, ma ora uso i tensori di torch e non numpy.
# f = w * x
# f = 2 * x

X = torch.tensor([1, 2, 3 , 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6 , 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad = True) #requires_grad aggiorna il gradiente quando faccio un'operazione con w

#model prediction
def forward(x): # l'operazione * funziona anche con i tensori, anche se non sono di python vanilla
    return w * x 

#loss = MSE
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()


print(f'Prediction before training: f(5) = {forward(5):.3f}')

#Training
learning_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
    #prediction = forward pass
    y_pred = forward(X) # nell'oggetto y_pred dovrebbe essere presente anche il gradiente rispetto a w

    #loss
    l = loss(Y, y_pred) # gradiente presente anche in loss

    #gradients = backward pass
    l.backward() # calcola d(loss)/dw (tenendo conto delle operazioni fatte per calcolare loss e y_pred) e aggiorna w.grad
    
    #update weights, modifica w, torch.no_grad() serve a non modificare il gradiente visto che in questo caso non occorre,
    # si sta aggiornando il peso e basta
    with torch.no_grad():
        w -= learning_rate * w.grad

    #zero gradients, il gradiente va azzerato, perché backward() non lo fa automaticamente, e si limita ad aggiungere a w.grad il gradiente appena calcolato
    w.grad.zero_()


    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')
