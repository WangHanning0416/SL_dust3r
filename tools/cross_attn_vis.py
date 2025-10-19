import os
import re
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import megfile
import glob

from bokeh.plotting import figure, curdoc
from bokeh.layouts import row, column
from bokeh.models import (ColumnDataSource, CustomJS, LinearColorMapper, HoverTool,
                          Select, Div)
from bokeh.palettes import Viridis256


# --------------------------
# 配置参数（请确保路径正确）
# --------------------------
NPY_DIR = "/data3/hanning/dust3r/cross_attn_npy/"  # 改为目录路径
SOURCE_IMG_PATH = "/data3/hanning/dust3r/tools/frame000001.png"
TARGET_IMG_PATH = "/data3/hanning/dust3r/tools/frame000000.png"
TEMPERATURE = 0.02
PATCH_SIZE = 14  # 14x14=196（根据npy形状调整）

info_div = Div(text="<h3>调试信息面板</h3><p>启动中...</p>")

def debug_print(message):
    """调试打印函数，同时输出到终端和页面信息面板"""
    print(f"[DEBUG] {message}")
    info_div.text += f"<br>[DEBUG] {message}"


def merge_attn_map(attn_maps, suppress_1st_attn=False, attn_layers_adopted: list[int] = None):
    """合并多层多头注意力图（带日志）"""
    debug_print(f"进入merge_attn_map，输入长度：{len(attn_maps)}")
    try:
        attn_maps = attn_maps if attn_layers_adopted is None else [attn_maps[idx] for idx in attn_layers_adopted]
        debug_print(f"筛选后层数：{len(attn_maps)}，单一层形状：{attn_maps[0].shape}")
        
        attn_maps = torch.stack(attn_maps, dim=1)
        debug_print(f"堆叠后形状：{attn_maps.shape}")
        
        attn_maps = torch.mean(attn_maps, dim=(1, 2))
        debug_print(f"合并层和头后形状：{attn_maps.shape}")
        
        if suppress_1st_attn:
            attn_maps[:, :, 0] = attn_maps.min()
            debug_print("已抑制第一个token的注意力")
        return attn_maps
    except Exception as e:
        debug_print(f"merge_attn_map错误：{str(e)}")
        raise

def load_image_as_rgba(img_path):
    """
    加载图片并转换为Bokeh兼容的RGBA格式（已修正数据溢出和坐标系问题）
    """
    debug_print(f"开始加载图片：{img_path}")
    try:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图片文件不存在：{img_path}")
        
        with Image.open(img_path) as img:
            debug_print(f"PIL.Image 初始模式: {img.mode}，尺寸: {img.size}")
            img = img.convert('RGBA')
            
            img_np = np.array(img, dtype=np.uint8)
            
            # 垂直翻转以匹配Bokeh的左下角原点坐标系
            img_np = np.flipud(img_np)
            debug_print("已执行 np.flipud() 垂直翻转")

            # 先将类型转为 uint32 再进行位运算，防止溢出
            r = img_np[:, :, 0].astype(np.uint32)
            g = img_np[:, :, 1].astype(np.uint32)
            b = img_np[:, :, 2].astype(np.uint32)
            a = img_np[:, :, 3].astype(np.uint32)
            
            rgba_array = (r << 24 | g << 16 | b << 8 | a)
            
            debug_print(f"最终 uint32 数组形状: {rgba_array.shape}, 数据类型: {rgba_array.dtype}")
            debug_print(f"uint32 数组 最小值: {np.min(rgba_array)}, 最大值: {np.max(rgba_array)}")
            
            return rgba_array, img.size[0], img.size[1]
            
    except Exception as e:
        debug_print(f"图片加载/转换错误: {str(e)}")
        raise

def softmax(x):
    """数值稳定的softmax"""
    try:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    except Exception as e:
        debug_print(f"softmax计算错误：{str(e)}")
        raise

def flip_attn_map(attn_map):
    """翻转注意力图（带日志）"""
    debug_print(f"翻转前注意力图形状：{attn_map.shape}")
    flipped = np.flipud(attn_map)
    debug_print(f"翻转后注意力图形状：{flipped.shape}")
    return flipped

if not os.path.isdir(NPY_DIR):
    error_div = Div(text=f"<h3>错误：目录不存在！</h3><p>{NPY_DIR}</p>")
    curdoc().add_root(error_div)
else:
    # 收集layer_0到layer_11下的注意力图文件
    npy_files = []
    for layer_num in range(12):  # 遍历0-11层
        layer_dir = os.path.join(NPY_DIR, f"layer_{layer_num}")
        attn_file = os.path.join(layer_dir, "img1_to_img2_attn.npy")  # 假设文件名固定
        if os.path.exists(attn_file):
            rel_path = os.path.relpath(attn_file, NPY_DIR)
            npy_files.append(rel_path)
            debug_print(f"找到有效文件：{rel_path}")
        else:
            debug_print(f"警告：未找到文件 {attn_file}")

    if not npy_files:
        error_div = Div(text=f"<h3>错误：未找到符合条件的npy文件</h3><p>目录：{NPY_DIR}</p>")
        curdoc().add_root(error_div)
    else:
        npy_files.sort(key=lambda x: int(re.search(r"layer_(\d+)", x).group(1)))
        data_dir = NPY_DIR  # 数据根目录

        # 全局变量
        similarity_array = None
        temp = TEMPERATURE
        src_width, src_height = 0, 0  # 源图片尺寸
        trg_width, trg_height = 0, 0  # 目标图片尺寸

        # 创建UI组件
        file_selector = Select(
            title="注意力图文件 (layer_0到layer_11):",
            value=npy_files[0],
            options=npy_files
        )
        trg_source = ColumnDataSource(data={'image': []})
        src_source = ColumnDataSource(data={'image': []})
        heatmap_source = ColumnDataSource(data={'x': [], 'y': [], 'similarity': []})
        max_point_source = ColumnDataSource(data={'x': [], 'y': []})
        trg_hover_source = ColumnDataSource(data={'x': [], 'y': []})
        signal_source = ColumnDataSource(data={'x': [], 'y': []})
        trg_selected_dot_source = ColumnDataSource(data={'x': [], 'y': []})

        # 创建图表（不预设尺寸，后续根据图片原始尺寸设置）
        hover_tool = HoverTool(tooltips=None, mode='mouse')
        p_trg = figure(tools=[hover_tool, 'tap', 'pan', 'wheel_zoom', 'reset'],
                      title="目标图片", match_aspect=True)
        p_src = figure(tools="pan,wheel_zoom,reset,help",
                      title="源图片", match_aspect=True)

        # 渲染图片（初始设置为临时尺寸）
        trg_image_renderer = p_trg.image_rgba(image='image', x=0, y=0, dw=1, dh=1, source=trg_source)
        src_image_renderer = p_src.image_rgba(image='image', x=0, y=0, dw=1, dh=1, source=src_source)

        # 目标图片交互
        p_trg.grid.grid_line_color = None
        p_trg.rect(x='x', y='y', width=1, height=1, source=trg_hover_source, fill_alpha=0,
                  line_color=None, hover_fill_alpha=0.3, hover_fill_color='white')
        p_trg.js_on_event('tap', CustomJS(args={'source': signal_source}, 
                                         code="source.data = {x: [cb_obj.x], y: [cb_obj.y]};"))
        p_trg.circle(x='x', y='y', size=10, color='red', source=trg_selected_dot_source)
        
        # 源图片热力图
        p_src.grid.grid_line_color = None
        color_mapper = LinearColorMapper(palette=Viridis256, low=0, high=1)
        heatmap_renderer = p_src.rect(x='x', y='y', width=1, height=1,
                                     fill_color={'field': 'similarity', 'transform': color_mapper},
                                     fill_alpha=0.5, line_color=None, source=heatmap_source)
        p_src.circle(x='x', y='y', size=10, color='red', source=max_point_source)

        def update_visualization(filepath):
            global similarity_array, src_width, src_height, trg_width, trg_height
            try:
                info_div.text = "<h3>调试信息面板</h3><p>开始加载数据...</p>"
                name = os.path.basename(filepath)
                debug_print(f"处理文件：{name}，路径：{filepath}")

                # 1. 加载npy文件
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"npy文件不存在：{filepath}")

                with megfile.smart_open(filepath, 'rb') as f:
                    attn_np = np.load(f)

                attn_tensor = torch.from_numpy(attn_np)
                attn_maps_list = [attn_tensor]
                merged_attn = merge_attn_map(attn_maps_list, suppress_1st_attn=False)

                merged_attn_np = merged_attn.cpu().numpy()
                debug_print(f"转为NumPy后形状：{merged_attn_np.shape}")
                expected_shape = (1, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)
                if merged_attn_np.size != np.prod(expected_shape):
                    raise ValueError(f"reshape失败：原始大小{merged_attn_np.size} != 预期大小{np.prod(expected_shape)}")
                similarity_array = merged_attn_np.reshape(expected_shape)[0]
                similarity_array = flip_attn_map(similarity_array)
                debug_print(f"最终注意力图形状：{similarity_array.shape}")

                # 4. 加载图片（关键：不修改尺寸！）
                src_img, src_width, src_height = load_image_as_rgba(SOURCE_IMG_PATH)  # 获取宽度和高度
                trg_img, trg_width, trg_height = load_image_as_rgba(TARGET_IMG_PATH)
                debug_print(f"源图片原始尺寸：宽={src_width}，高={src_height}")
                debug_print(f"目标图片原始尺寸：宽={trg_width}，高={trg_height}")

                # 5. 更新图片数据源（使用原始尺寸）
                trg_source.data = {'image': [trg_img]}
                src_source.data = {'image': [src_img]}
                debug_print("图片数据已更新到数据源（原始尺寸）")

                # 6. 设置图表坐标范围为图片原始像素范围（0~宽，0~高）
                p_trg.x_range.start, p_trg.x_range.end = 0, trg_width
                p_trg.y_range.start, p_trg.y_range.end = 0, trg_height
                p_src.x_range.start, p_src.x_range.end = 0, src_width
                p_src.y_range.start, p_src.y_range.end = 0, src_height
                debug_print(f"目标图坐标范围：x=0~{trg_width}，y=0~{trg_height}")
                debug_print(f"源图坐标范围：x=0~{src_width}，y=0~{src_height}")

                # 7. 设置图片显示尺寸为原始尺寸（dw=宽，dh=高）
                trg_image_renderer.glyph.dw, trg_image_renderer.glyph.dh = trg_width, trg_height
                src_image_renderer.glyph.dw, src_image_renderer.glyph.dh = src_width, src_height
                debug_print(f"目标图显示尺寸：dw={trg_width}，dh={trg_height}")
                debug_print(f"源图显示尺寸：dw={src_width}，dh={src_height}")

                # 8. 图表大小适配图片（最大不超过800px，避免过大）
                MAX_SIZE = 800
                trg_scale = min(MAX_SIZE / trg_width, MAX_SIZE / trg_height, 1.0)
                src_scale = min(MAX_SIZE / src_width, MAX_SIZE / src_height, 1.0)
                trg_display_w, trg_display_h = int(trg_width * trg_scale), int(trg_height * trg_scale)
                src_display_w, src_display_h = int(src_width * src_scale), int(src_height * src_scale)
                p_trg.width, p_trg.height = trg_display_w, trg_display_h
                p_src.width, p_src.height = src_display_w, src_display_h
                debug_print(f"目标图表显示尺寸：{trg_display_w}x{trg_display_h}")
                debug_print(f"源图表显示尺寸：{src_display_w}x{src_display_h}")

                # 9. 更新标题
                p_trg.title.text = f"目标图片：{Path(TARGET_IMG_PATH).name}（{trg_width}x{trg_height}）"
                p_src.title.text = f"源图片：{Path(SOURCE_IMG_PATH).name}（{src_width}x{src_height}）"

                # 10. 计算patch尺寸（基于原始图片尺寸）
                h_attn, w_attn = similarity_array.shape[0:2]
                patch_size_h, patch_size_w = trg_height / h_attn, trg_width / w_attn  # 每个patch的像素大小
                debug_print(f"patch尺寸：高={patch_size_h:.2f}px，宽={patch_size_w:.2f}px")

                # 11. 初始化热力图网格（每个patch的中心坐标）
                patch_x = np.tile(np.arange(w_attn) * patch_size_w + patch_size_w/2, h_attn)
                patch_y = np.repeat(
                    (h_attn - 1 - np.arange(h_attn)) * patch_size_h + patch_size_h/2, 
                    w_attn
                )
                trg_hover_source.data = {'x': patch_x, 'y': patch_y}
                heatmap_source.data['x'], heatmap_source.data['y'] = patch_x, patch_y
                heatmap_renderer.glyph.width, heatmap_renderer.glyph.height = patch_size_w, patch_size_h
                debug_print(f"热力图网格初始化完成，patch数量：{len(patch_x)}")

                # 12. 初始化热力图
                update_heatmap(None, None, {'x': [trg_width/2], 'y': [trg_height/2]})  # 从图片中心开始
                debug_print("初始化完成，可视化准备就绪")

            except Exception as e:
                debug_print(f"加载失败：{str(e)}")
                raise


        def update_heatmap(attr, old, new):
            global temp
            try:
                if not new.get('x') or similarity_array is None:
                    debug_print("update_heatmap：无输入或注意力图未加载")
                    return
                
                # 计算点击位置对应的patch索引
                H, W = trg_height, trg_width
                h_attn, w_attn = similarity_array.shape[0:2]
                patch_size_h, patch_size_w = H / h_attn, W / w_attn
                x_click, y_click = new['x'][0], new['y'][0]
                i = max(0, min(h_attn - 1, int(y_click / patch_size_h)))
                j = max(0, min(w_attn - 1, int(x_click / patch_size_w)))
                debug_print(f"点击位置：({x_click:.1f}, {y_click:.1f})，对应patch索引：i={i}, j={j}")

                # 更新目标图片选中标记
                selected_x = j * patch_size_w + patch_size_w/2
                selected_y = i * patch_size_h + patch_size_h/2
                trg_selected_dot_source.data = {'x': [selected_x], 'y': [selected_y]}

                # 提取并归一化注意力值
                new_similarity = similarity_array[i, j, :, :].flatten()
                debug_print(f"提取的注意力值数量：{len(new_similarity)}")
                if temp > 0:
                    new_similarity = softmax(new_similarity / temp)
                else:
                    new_similarity = (new_similarity - new_similarity.min()) / (new_similarity.max() - new_similarity.min() + 1e-8)
                heatmap_source.data['similarity'] = new_similarity

                # 标记最大注意力位置
                max_idx = np.argmax(new_similarity)
                patch_x, patch_y = heatmap_source.data['x'], heatmap_source.data['y']
                max_point_source.data = {'x': [patch_x[max_idx]], 'y': [patch_y[max_idx]]}

            except Exception as e:
                debug_print(f"update_heatmap错误：{str(e)}")


        def on_file_select(attr, old, new):
            debug_print(f"切换文件：{new}")
            # 拼接完整路径（数据目录 + 相对路径）
            full_path = os.path.join(data_dir, new)
            update_visualization(full_path)


        # --------------------------
        # 绑定回调和布局
        # --------------------------
        file_selector.on_change('value', on_file_select)
        signal_source.on_change('data', update_heatmap)
        controls = column(file_selector, info_div, width=400)
        plots = row(p_trg, p_src)
        main_layout = row(controls, plots)
        curdoc().add_root(main_layout)
        curdoc().title = "注意力图可视化（layer_0到layer_11）"
        
        # 初始化显示第一个文件
        debug_print("启动初始化...")
        update_visualization(os.path.join(data_dir, npy_files[0]))