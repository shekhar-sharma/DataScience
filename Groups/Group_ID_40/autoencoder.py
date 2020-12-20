from torch import nn, cat
from torch.autograd import Variable

#ZDIMS is the bottleneck dimensions, i.e the dimensions of our latent space
ZDIMS=20
class Autoencoder(nn.Module):
    def __init__(self,Zdims,input_dim):
        super(Autoencoder, self).__init__()
        # ENCODER
        self.input_dim=input_dim
        ZDIMS=Zdims
        self.en_z_1 = nn.Linear(input_dim, 1024)
        self.en_z_2 = nn.Linear(1024, 1024)
        self.en_z_3 = nn.Linear(1024, 1024)

        # rectified linear unit layer from 400 to 400
        # max(0, x)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.en_z_4_mu = nn.Linear(1024, ZDIMS)  # mu layer
        self.en_z_4_sigma = nn.Linear(1024, ZDIMS)  # logvariance layer
        # this last layer bottlenecks through ZDIMS connections
        self.de_x_1 = nn.Linear(ZDIMS, 1024)
        self.de_x_2 = nn.Linear(1024, 1024)
        self.de_x_3 = nn.Linear(1024, 1024)
        self.de_x_4 = nn.Linear(1024, self.input_dim)

        # DECODER 2
        self.de_y_1 = nn.Linear(ZDIMS, 1024)
        self.de_y_2 = nn.Linear(1024, 1024)
        self.de_y_3 = nn.Linear(1024, 1024)
        self.de_y_4 = nn.Linear(1024, self.input_dim)

        self.sigmoid = nn.Sigmoid()

    def encode(self, x: Variable) -> (Variable, Variable):
        """Input vector x -> fully connected 1 -> ReLU -> (fully connected

        """
        h1 = self.relu(self.en_z_1(self.dropout(x)))
        h1 = self.relu(self.en_z_2(self.dropout(h1)))
        h1 = self.relu(self.en_z_3(self.dropout(h1)))
        return self.en_z_4_mu(self.dropout(h1)), self.en_z_4_sigma(self.dropout(h1))

    def reparameterize(self, mu: Variable, logvar: Variable) -> Variable:
        """THE REPARAMETERIZATION IDEA:

        """

        if self.training:

            std = logvar.mul(0.5).exp_()  # type: Variable

            eps = Variable(std.data.new(std.size()).normal_())

            return eps.mul(std).add_(mu)

        else:
            # During inference, we simply spit out the mean of the
            # learned distribution for the current input.  We could
            # use a random sample from the distribution, but mu of
            # course has the highest probability.
            return mu

    def decode_1(self, z: Variable) -> Variable:
        h3 = self.relu(self.de_x_1(self.dropout(z)))
        h3 = self.relu(self.de_x_2(self.dropout(h3)))
        h3 = self.relu(self.de_x_3(self.dropout(h3)))
        return self.sigmoid(self.de_x_4(self.dropout(h3)))

    def decode_2(self, z: Variable) -> Variable:
        h3 = self.relu(self.de_y_1(self.dropout(z)))
        h3 = self.relu(self.de_y_2(self.dropout(h3)))
        h3 = self.relu(self.de_y_3(self.dropout(h3)))
        return self.sigmoid(self.de_y_4(self.dropout(h3)))
    
    def forward(self, x: Variable, y: Variable) -> (Variable, Variable, Variable):
        mu, log_var = self.encode(x.view(-1, self.input_dim))
        z1 = self.reparameterize(mu, log_var)
        z2 = self.reparameterize(mu, log_var)
        return self.decode_1(z1), self.decode_2(z2), mu, log_var
    


