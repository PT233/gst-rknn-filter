plugin_c_args = ['-DHAVE_CONFIG_H']

cdata = configuration_data()
cdata.set_quoted('PACKAGE_VERSION', gst_version)
cdata.set_quoted('PACKAGE', 'gst-plugin-rknn')
cdata.set_quoted('GST_LICENSE', 'LGPL')
cdata.set_quoted('GST_API_VERSION', api_version)
cdata.set_quoted('GST_PACKAGE_NAME', 'GStreamer RKNN Plug-in')
cdata.set_quoted('GST_PACKAGE_ORIGIN', 'https://gstreamer.freedesktop.org')
configure_file(output : 'config.h', configuration : cdata)

# Plugin
plugin_sources = [
  'gstrknn.c',
  'rgaprocess.cc',
  'dmabuffer.c',
  'rknnprocess.cc',
  'postprocess.cc',
  'postprocess/postprocess_dispatcher.cc',
  'postprocess/postprocess_yolov8.cc',
  'postprocess/postprocess_yolov6.cc',
  'postprocess/postprocess_classification.cc',
  'postprocess/postprocess_retinaface.cc',
  ]

postprocess_inc = include_directories('.', 'postprocess')

library('gstrknn',
  plugin_sources,
  c_args: plugin_c_args,
  include_directories : postprocess_inc,
  dependencies : [gst_dep, gstbase_dep, gstallocators_dep, gstvideo_dep, gstpbutils_dep, rga_dep, rknn_dep, rga_dep, opencv_dep],
  install : true,
  install_dir : plugins_install_dir,
)
