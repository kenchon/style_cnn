import stylenet

class LinearUnit(nn.Module):
    def __init__(self, in_size, out_size, activation_type):
        super(LinearUnit, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.linear = nn.Linear(in_size, out_size)
        if activation_type == "sigmoid":
            self.activation_f = nn.Sigmoid()
        elif activation_type == "softmax":
            self.activation_f = nn.Softmax()

    def forward(self, input):
        x = self.linear(input)
        x = self.activation_f(x)
        return x

def train(epoch, optim):
    for epoch in range(epochs):

        runnning_loss = 0.0
        for i


if __name__ == "__main__":

    cnn = stylenet.Stylenet()
    linearA = LinearUnit(128, 5,  "softmax")    # linear unit for style classification
    linearB = LinearUnit(128, 66, "sigmoid")    # linear unit for weak label prediction

    cnn.load_state_dict(torch.load("linear_weight_softmax_pro_ver_162000_.pth"))

    model.cuda()
    model.train()

    # learning settings
    learning_rate = 5e-3
    epochs = 100
    optimizer = optim.SGD([
                    {'params': model.parameters(), 'lr': 1e-5},
                    {'params': unitA.parameters()},
                    {'params': unitB.parameters()}
                    ], lr=1e-3)
    batch_size = 16
    #alpha_c = 0.01
    #split = 1000

    #max_score = 0
    loss_progress = []

    for epoch in range(epochs):

        row, batchs = load_training_ids(batch_size, use_proposed = False)
        print(len(batchs))

        for o, batch in enumerate(batchs):
            loss = 0
            model.train()

            if(o% split == 0 and o != 0):
                model_path = "./result/params/prams_lr001_clas=False_pre_epoch{}_iter{}_12.pth".format(epoch, o)
                optim_path = "./result/params/optim_lr001_clas=False_pre_epoch{}_iter{}_12.pth".format(epoch, o)
                torch.save(model.state_dict(), model_path)
                torch.save(optimizer.state_dict(), optim_path)

                temp_score = test.test(model_path, do_classification, model)

                path_to_fig = get_loss_acc_fig(loss_progress, temp_score)
                ln.send_image_(path_to_fig, 'loss progress')
                ln.notify('{} {}'.format(str(temp_score), model_path))

                loss_progress = []

            # img, sim are list: i th column holds i th triplet
            img, sim, target, target_npy = triplet_sampling(row, batch)
            batch_count = 0

            feat = []
            pred = []
            for j in range(3):
                f= forward(Variable(img[j]).cuda())
                if(do_classification and j == 2):
                    pred.append(p)

                feat.append(f)

            # LOSS COMPUTING
            for i in range(batch_size):
                loss += triplet_loss([feat[0][i],feat[1][i],feat[2][i]], (sim[0][i],sim[1][i]), use_proposed=False)
                if(do_classification):
                    loss += alpha_c * binary_classfi_loss(pred[0][i], target_npy[i], use_proposed=True)

            loss = loss/batch_size
            print("{} {:.4f}".format(o, loss.cpu().detach().numpy()))
            loss_progress.append("{:.4f}".format(loss.cpu().detach().numpy()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
