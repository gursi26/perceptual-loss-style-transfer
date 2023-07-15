import torch 
from torchvision.models import vgg16
from torchvision import transforms 


# Loss network 
class vgg(torch.nn.Module):

    def __init__(self):
        super(vgg,self).__init__()

        self.activation_layers = [3,8,15,22]
        self.model = vgg16(pretrained=True).features[:23]

    def forward(self,x):
        activations = []
        for layer_number, layer in enumerate(self.model):
            x = layer(x)
            if layer_number in self.activation_layers : 
                activations.append(x)

        return activations

def test_vgg():
    model = vgg()
    x = torch.randn(1,3,256,256)
    out = model(x)
    print(x.shape)
    for i in out :
        print(i.shape)

#test_vgg()


# Full loss function
class LossFn() : 
     
    def __init__(self, style_image, content_weight, style_weight, batch_size, dev) : 
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.dev = dev
        self.mse = torch.nn.MSELoss()
        self.loss_network = vgg().to(self.dev)

        self.style = style_image.unsqueeze(0)
        self.style_activations = self.loss_network(self.vgg_normalize(self.style))
        for i in range(len(self.style_activations)) :
            self.style_activations[i] = self.style_activations[i].repeat(batch_size,1,1,1)
        self.style_grams = [self.gram_matrix(x) for x in self.style_activations]


    def vgg_normalize(self, x) :
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        x = x.div_(255.0)[:, 0:3, :, :]
        return normalize(x)


    def content_loss(self, output, content):
        loss = self.mse(output, content)
        return loss * self.content_weight


    def gram_matrix(self, x):
        (batch, ch, h, w) = x.size() 
        img = x.view(batch, ch, h*w)
        img_transposed = img.transpose(1,2)
        gram_matrix = img.bmm(img_transposed) / (ch * h * w)
        return gram_matrix


    def style_loss(self, output_activations):
        style_loss = 0 
        for out_ac, style_gram_matrix in zip(output_activations, self.style_grams):
            output_gram_matrix = self.gram_matrix(out_ac)
            style_loss += self.mse(output_gram_matrix, style_gram_matrix)
        return style_loss * self.style_weight


    def calc_loss(self, output, content, content_loss_layer=0):
        content_activations = self.loss_network(self.vgg_normalize(content))
        output_activations = self.loss_network(self.vgg_normalize(output))

        content_loss = self.content_loss(output_activations[content_loss_layer], content_activations[content_loss_layer])
        style_loss = self.style_loss(output_activations)
        total_loss = content_loss + style_loss
        return total_loss, content_loss, style_loss


def test_loss():
    style_image = torch.randn(3,256,256)
    content_image = torch.randn(8,3,256,256)
    output = torch.randn(8,3,256,256)
    content_weight = 1
    style_weight = 3

    crit = LossFn(style_image, content_weight, style_weight, 8, 'cpu')
    loss = crit.calc_loss(output, content_image)

    print(f'style shape : {style_image.shape}')
    print(f'content shape : {content_image.shape}')
    print(f'output shape : {output.shape}')
    print(f'loss : {loss.item()}')

# test_loss()