import numpy as np
import os
import ntpath
import time
import matplotlib.pyplot as plt
from . import util
from . import html

# class Visualizer():
#     def __init__(self, opt):
#         # self.opt = opt
#         self.display_id = opt.display_id
#         self.use_html = opt.isTrain and not opt.no_html
#         self.win_size = opt.display_winsize
#         self.name = opt.name
#         if self.display_id > 0:
#             import visdom
#             self.vis = visdom.Visdom(port = opt.display_port)
#             self.display_single_pane_ncols = opt.display_single_pane_ncols

#         if self.use_html:
#             self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
#             self.img_dir = os.path.join(self.web_dir, 'images')
#             print('create web directory %s...' % self.web_dir)
#             util.mkdirs([self.web_dir, self.img_dir])
#         self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
#         with open(self.log_name, "a") as log_file:
#             now = time.strftime("%c")
#             log_file.write('================ Training Loss (%s) ================\n' % now)

#     # |visuals|: dictionary of images to display or save
#     def display_current_results(self, visuals, epoch):
#         if self.display_id > 0: # show images in the browser
#             if self.display_single_pane_ncols > 0:
#                 h, w = next(iter(visuals.values())).shape[:2]
#                 table_css = """<style>
#     table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
#     table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
# </style>""" % (w, h)
#                 ncols = self.display_single_pane_ncols
#                 title = self.name
#                 label_html = ''
#                 label_html_row = ''
#                 nrows = int(np.ceil(len(visuals.items()) / ncols))
#                 images = []
#                 idx = 0
#                 for label, image_numpy in visuals.items():
#                     label_html_row += '<td>%s</td>' % label
#                     images.append(image_numpy.transpose([2, 0, 1]))
#                     idx += 1
#                     if idx % ncols == 0:
#                         label_html += '<tr>%s</tr>' % label_html_row
#                         label_html_row = ''
#                 white_image = np.ones_like(image_numpy.transpose([2, 0, 1]))*255
#                 while idx % ncols != 0:
#                     images.append(white_image)
#                     label_html_row += '<td></td>'
#                     idx += 1
#                 if label_html_row != '':
#                     label_html += '<tr>%s</tr>' % label_html_row
#                 # pane col = image row
#                 self.vis.images(images, nrow=ncols, win=self.display_id + 1,
#                                 padding=2, opts=dict(title=title + ' images'))
#                 label_html = '<table>%s</table>' % label_html
#                 self.vis.text(table_css + label_html, win = self.display_id + 2,
#                               opts=dict(title=title + ' labels'))
#             else:
#                 idx = 1
#                 for label, image_numpy in visuals.items():
#                     #image_numpy = np.flipud(image_numpy)
#                     self.vis.image(image_numpy.transpose([2,0,1]), opts=dict(title=label),
#                                        win=self.display_id + idx)
#                     idx += 1

#         if self.use_html: # save images to a html file
#             for label, image_numpy in visuals.items():
#                 img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
#                 util.save_image(image_numpy, img_path)
#             # update website
#             webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
#             for n in range(epoch, 0, -1):
#                 webpage.add_header('epoch [%d]' % n)
#                 ims = []
#                 txts = []
#                 links = []

#                 for label, image_numpy in visuals.items():
#                     img_path = 'epoch%.3d_%s.png' % (n, label)
#                     ims.append(img_path)
#                     txts.append(label)
#                     links.append(img_path)
#                 webpage.add_images(ims, txts, links, width=self.win_size)
#             webpage.save()

#     # errors: dictionary of error labels and values
#     def plot_current_errors(self, epoch, counter_ratio, opt, errors):
#         if not hasattr(self, 'plot_data'):
#             self.plot_data = {'X':[],'Y':[], 'legend':list(errors.keys())}
#         self.plot_data['X'].append(epoch + counter_ratio)
#         self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
#         self.vis.line(
#             X=np.stack([np.array(self.plot_data['X'])]*len(self.plot_data['legend']),1),
#             Y=np.array(self.plot_data['Y']),
#             opts={
#                 'title': self.name + ' loss over time',
#                 'legend': self.plot_data['legend'],
#                 'xlabel': 'epoch',
#                 'ylabel': 'loss'},
#             win=self.display_id)

#     # errors: same format as |errors| of plotCurrentErrors
#     def print_current_errors(self, epoch, i, errors, t):
#         message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
#         for k, v in errors.items():
#             message += '%s: %.3f ' % (k, v)

#         print(message)
#         with open(self.log_name, "a") as log_file:
#             log_file.write('%s\n' % message)

#     # save image to the disk
#     def save_images(self, webpage, visuals, image_path):
#         image_dir = webpage.get_image_dir()
#         short_path = ntpath.basename(image_path[0])
#         name = os.path.splitext(short_path)[0]

#         webpage.add_header(name)
#         ims = []
#         txts = []
#         links = []

#         for label, image_numpy in visuals.items():
#             image_name = '%s_%s.png' % (name, label)
#             save_path = os.path.join(image_dir, image_name)
#             util.save_image(image_numpy, save_path)

#             ims.append(image_name)
#             txts.append(label)
#             links.append(image_name)
#         webpage.add_images(ims, txts, links, width=self.win_size)


#     def save_images_demo(self, webpage, visuals, image_path):
#         image_dir = webpage.get_image_dir()
#         short_path = ntpath.basename(image_path[0])
#         name = os.path.splitext(short_path)[0]

#         webpage.add_header(name)
#         ims = []
#         txts = []
#         links = []

#         for label, image_numpy in visuals.items():
#             image_name = '%s.jpg' % (name)
#             save_path = os.path.join(image_dir, image_name)
#             util.save_image(image_numpy, save_path)

#             ims.append(image_name)
#             txts.append(label)
#             links.append(image_name)
#         webpage.add_images(ims, txts, links, width=self.win_size)



import os
import time
import ntpath
from .util import save_image  # 假设 save_image 是一个保存图像的实用函数

class Visualizer:
    def __init__(self, opt):
        self.display_id = opt.display_id
        self.name = opt.name
        self.output_dir = os.path.join(opt.outputs_dir,opt.name)  # 指定输出目录

        # 创建输出目录（若不存在）
        self.img_dir = os.path.join(self.output_dir, 'images')
        os.makedirs(self.img_dir, exist_ok=True)

        # 日志文件
        self.log_name = os.path.join(self.output_dir, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write(f'================ Training Loss ({now}) ================\n')

    # 保存当前结果到本地
    def save_current_results(self, visuals, epoch):
        """
        保存视觉化结果为图片。
        :param visuals: 一个包含图像的字典，键为图像标签，值为图像（numpy数组）。
        :param epoch: 当前的 epoch 编号。
        """
        for label, image_numpy in visuals.items():
            img_path = os.path.join(self.img_dir, f'epoch{epoch:03d}_{label}.png')
            save_image(image_numpy, img_path)

    # 记录当前错误到日志
    def log_current_errors(self, epoch, i, errors, t):
        """
        打印并保存当前训练的错误信息。
        :param epoch: 当前的 epoch 编号。
        :param i: 当前的迭代次数。
        :param errors: 一个包含错误的字典，键为错误标签，值为错误值。
        :param t: 当前迭代的用时。
        """
        message = f'(epoch: {epoch}, iters: {i}, time: {t:.3f}) '
        message += ' '.join([f'{k}: {v:.3f}' for k, v in errors.items()])
        print(message)

        with open(self.log_name, "a") as log_file:
            log_file.write(f'{message}\n')

    # 保存图片（带特定文件路径）
    def save_images(self, visuals, image_path):
        """
        保存图像到本地，文件名基于输入路径生成。
        :param visuals: 一个包含图像的字典，键为图像标签，值为图像（numpy数组）。
        :param image_path: 图像路径，用于生成保存文件名。
        """
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        for label, image_numpy in visuals.items():
            image_name = f'{name}_{label}.png'
            save_path = os.path.join(self.img_dir, image_name)
            save_image(image_numpy, save_path)


    def display_current_results(self, visuals, epoch):
        # Ensure directory for saving images exists
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
            
        # Save images to local directory
        for label, image_numpy in visuals.items():
            img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
            util.save_image(image_numpy, img_path)

        # Optional: Print message to confirm saving
        print(f"Images for epoch {epoch} saved to {self.img_dir}")



    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)


    def plot_current_errors(self, epoch, counter_ratio, opt, errors):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}

        # Add new data point to the plot data
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])

        # Create the plot
        plt.figure(figsize=(10, 5))
        
        # Plot each error in the dictionary
        for i, label in enumerate(self.plot_data['legend']):
            plt.plot(self.plot_data['X'], np.array(self.plot_data['Y'])[:, i], label=label)
        
        # Add labels and title
        plt.title(self.name + ' loss over time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='best')

        # Save the plot to a local file
        plot_dir = os.path.join(opt.checkpoints_dir, 'plots')  # Make sure to create a "plots" folder in the checkpoints directory
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"epoch_{epoch:03d}_loss.png")
        plt.savefig(plot_path)
        plt.close()