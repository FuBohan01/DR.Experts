import torch
import torch.nn as nn
import torch.nn.functional as F

class Dice(nn.Module):
    """
    自定义的dice激活函数
    """
    def __init__(self):
        super(Dice, self).__init__()
        self.alpha = nn.Parameter(torch.zeros((1,)))
        self.epsilon = 1e-9
    
    def forward(self, x):

        norm_x = (x - x.mean(dim=0)) / torch.sqrt(x.var(dim=0) + self.epsilon)
        p = torch.sigmoid(norm_x)
        x = self.alpha * x.mul(1-p) + x.mul(p)
    
        return x
    
class ActivationUnit(nn.Module):
    """
    激活函数单元
    功能是计算用户购买行为与推荐目标之间的注意力系数，比如说用户虽然用户买了这个东西，但是这个东西实际上和推荐目标之间没啥关系，也不重要，所以要乘以一个小权重
    """
    def __init__(self, embedding_dim, fc_dims = [384, 32, 16]):
        super(ActivationUnit, self).__init__()
        # 1.初始化fc层
        fc_layers = []
        # 2.输入特征维度
        input_dim = embedding_dim*3     
        # 3.fc层内容：全连接层（4*embedding,32）—>激活函数->全连接层（32,16）->.....->全连接层（16,1）
        for fc_dim in fc_dims:
            fc_layers.append(nn.Linear(input_dim, fc_dim))
            fc_layers.append(nn.PReLU())
            input_dim = fc_dim
        
        fc_layers.append(nn.Linear(input_dim, 1))
        # 4.将上面定义的fc层，整合到sequential中
        self.fc = nn.Sequential(*fc_layers)
    
    def forward(self, query, user_behavior):
        """
            :param query:targe目标的embedding ->（输入维度） batch*1*embed 
            :param user_behavior:行为特征矩阵 ->（输入维度） batch*seq_len*embed
            :return out:预测目标与历史行为之间的注意力系数
        """
        # 1.获取用户历史行为序列长度
        # 3.前面的把四个embedding合并成一个（4*embedding）的向量，
        #  第一个向量是目标商品的向量，第二个向量是用户行为的向量，
        #  至于第三个和第四个则是他们的相减和相乘（这里猜测是为了添加一点非线性数据用于全连接层，充分训练）
        attn_input = torch.cat([query, user_behavior, query - user_behavior], dim = -1) # [bs, 10, 577, 4*384]
        out = self.fc(attn_input)
        return out
class AttentionPoolingLayer(nn.Module):
    """
      注意力序列层
      功能是计算用户行为与预测目标之间的系数，并将所有的向量进行相加，这里的目的是计算出用户的兴趣的能力向量
    """
    def __init__(self, embedding_dim):
        super(AttentionPoolingLayer, self).__init__()
        self.active_unit = ActivationUnit(embedding_dim = embedding_dim)
        # self.Linear = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, query_ad, user_behavior):
        """
          :param query_ad:targe目标x的embedding   -> （输入维度） batch*1*embed
          :param user_behavior:行为特征矩阵     -> （输入维度） batch*seq_len*embed
          :return output:用户行为向量之和，反应用户的爱好
        """
        # 1.计算目标和历史行为之间的相关性
        attns = self.active_unit(query_ad, user_behavior) 

        # user_behavior = self.Linear(user_behavior)  # [bs, 10, 577, 384]

        # 2.注意力系数乘以行为 
        output = user_behavior.mul(attns)
        # output = user_behavior.mul(query_ad)
        # 3.历史行为向量相加
        # output = user_behavior.sum(dim=1)
        output = output.sum(dim=1)
        return output
    
class DeepInterestNet(nn.Module):
    """
      模型主体
    """

    def __init__(self, embed_dim, class_num=10):
        super(DeepInterestNet, self).__init__()

        self.class_num = class_num

        # 3.注意力计算层（论文核心）
        self.AttentionActivate = AttentionPoolingLayer(embed_dim)
        # 5.该层的输入为历史行为的embedding，和目标的embedding，所以输入维度为2*embedding_dim
        #  全连接层（2*embedding,fc_dims[0]）—>激活函数->dropout->全连接层（fc_dims[0],fc_dims[1]）->.....->全连接层（fc_dims[n],1）   
    
    def forward(self, diff, img): # [bs, 10, 577, 384]  [bs, 577, 384]
        img = img.unsqueeze(1)
        img = img.expand(-1, self.class_num, -1, -1)

        
        user_interest = self.AttentionActivate(diff, img)
        # 8.将计算后的用户行为行为记录和推荐的目标进行拼接


        return user_interest

        
if __name__ == '__main__':
    x = torch.randn(2, 10, 577, 384)
    img = torch.randn(2, 577, 384)
    model = DeepInterestNet(384)
    out = model(x, img)
    print(out.shape)  # Should be [2, 64, 14, 8]
