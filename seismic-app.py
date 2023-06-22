import numpy as np
import dash
from dash import dcc, html
import dash_vtk
import random
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from utils import *

#----------------------------------------defaults ------------------------------
attribute = ['sweetness', 'infreq', 'reflin', 'rms', 'timegain', 'enve', 'fder', 'sder',
            'gradmag', 'inphase', 'cosphase', 'ampcontrast', 'ampacc', 'inband', 'domfreq',
            'resamp', 'apolar', 'resfreq', 'resphase'
            ]

noise = ['gaussian', 'median', 'convolution']

kernel = ['(10, 9, 1)', '(1, 1, 3)', '(3, 3, 1)', '(1, 1, 1)', 'None']

# cmap = ['erdc_rainbow_bright',  'Grayscale', 'Cold and Hot', 'Cool to Warm', 'Black, Blue and White',
#          'Black, Orange and White', 'Black-Body Radiation', 'Warm to Cool', 'X Ray', 'erdc_rainbow_dark',
#          'Rainbow Desaturated'
#          ]

cmap = ['puor', 'gray', 'phase', 'jet', 'plasma',
         'inferno', 'spectral', 'rainbow', 'electric']


#-------------------------------------------------------------
slice_property = {"colorWindow": 10, "colorLevel": 10}

#slice view
slice_view = dash_vtk.View(
    id="slice-view",
    cameraPosition=[1, 0, 0],
    cameraViewUp=[0, 0, -1],
    cameraParallelProjection=False,
    background=[0.5, 0.5, 1],
    children=[
        dash_vtk.SliceRepresentation(
            id="slice-repr-i",
            iSlice=25,
            property=slice_property,
            colorMapPreset="Black, Orange and White",
        ),
        dash_vtk.SliceRepresentation(
            id="slice-repr-j",
            jSlice=25,
            property=slice_property,
            colorMapPreset="Black, Orange and White",
        ),
        dash_vtk.SliceRepresentation(
            id="slice-repr-k",
            kSlice=25,
            property=slice_property,
            colorMapPreset="Black, Orange and White",
        ),
        dash_vtk.GeometryRepresentation(
            id='grid',
            children=[dash_vtk.PolyData(id='grid-points')],  
            property={"edgeVisibility": True, 'opacity':0.5, 
                      'color':(0, 0, 0)},
            showCubeAxes=True,
            cubeAxesStyle={"axisLabels": ["Z", "XL", "IL"]},
            colorMapPreset='X Ray'
        ),
    ],
)

#-------------------------------Component---------------------------------
sliders = {
    "Time Slice": dcc.Slider(id="slider-i", min=0, max=700, value=25),
    "Crossline": dcc.Slider(id="slider-j", min=0, max=700, value=25),
    "Inline": dcc.Slider(id="slider-k", min=0, max=700, value=25),
    "Color Level": dcc.Slider(id="slider-lvl", min=0, max=4000, value=1000),
    "Color Window": dcc.Slider(id="slider-window", min=0, max=4000, value=1000),
    'Attribute Type': dcc.Dropdown(attribute, id='attri-component', value='sweetness'),
    'Noise Algorithm': dcc.Dropdown(noise, id='noise-component', value='gaussian'),
    'Kernel':dcc.Dropdown(kernel, id='kernel-component', value='None'),
    "Threshold": dcc.Slider(id="threshold", min=-500, max=500, value=20),
    'Color Map': dcc.Dropdown(cmap, id='scale-component', value='spectral', multi=False)
    }

controls = dbc.Card(
    body=True,
    children=dbc.Row(
        [
            dbc.Col([dbc.Label(label), component], style={"width": "250px"})
            for label, component in sliders.items()
        ]
    ),
)

#-------------------------------------------------App layout--------------------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container(
    fluid=True,
    style={"height": "calc(100vh - 30px)"},
    children=[
        html.H1('Welcome to SeisAI!',
            style={'text-align':'center',
                    'color':'blue'}),
                    html.H2(html.Em('Making subsurface interpretation seemless'),
                            style={'text-align':'center'}),
                            html.Br(),
                            html.Div(
                         children=[
                            html.Div(dcc.Upload(
                                id='upload-seismic',
                                children=html.Button('Upload SEGY',
                                                     style={'backgroundColor': '#111100',
                                                             'color':'white',
                                                            'textAlign':'center'}),
                                multiple=False),
                                style={'width':'9%', 'height':'5%'},
                            )
                        ]),
        html.Br(),
        dbc.CardGroup(
        [
            dbc.Label(html.B("Display seismic: ")),
            dbc.Checklist(
                options=[{"label": "Time Slice", "value": "inline"}, 
                         {"label": "Crossline", "value": "xline"}, 
                         {'label':'Inline','value':'z'},
                         {'label':'Grid', 'value':'grid'}
                         ],
                value=["inline", "xline", 'z', 'grid'],
                id="enabled",
                inline=True,
            ),
        ]
        ),
        html.Div(
            style={"height": "20%", "display": "flex", "alignItems": "center"},
            children=[
                # html.Br(),
                controls,
                # html.Br(),
            ],
        ),
        html.Div([
            html.Button("Export Mask Volume", id="btn-segy",
                        style={'backgroundColor': '#111100',
                                'color':'white',
                                'textAlign':'center'
                                },
                           
            ),
            dcc.Download(id="download_segy", base64=True)
        ]),
        html.Br(),
        html.Div(
                dbc.Row([
                    dbc.Col(dcc.Graph(id="fig1", className="three columns")),
                    dbc.Col(dcc.Graph(id="fig2", className="three columns")),
                    dbc.Col(dcc.Graph(id="fig3", className="three columns")),
                    dbc.Col(dcc.Graph(id="fig4",  className="three columns"))
                ],
                style={"height": "20%", "display": "flex", "alignItems": "center"}
        )),
        html.Div(
                [
                    dbc.Row([
                        dbc.Col(
                            html.Button('Prev', id='btn-a'),
                            className='one columns',
                            style={'textAlign':'left'}
                    ),
                    dbc.Col(
                        html.Button('Next', id='btn-b'),
                        className='one columns',
                        style={'textAlign':'right'}
                    )
                ], className='row'
            )
                ]
        ),
        html.Br(),
        html.Div(slice_view, style={"height": "80%"}), 
        dcc.Store(id='segyfile')
    ],
)

#----------------------Callbacks---------------------------
@app.callback(
        [
            Output('slice-repr-i', 'children'),
            Output('slice-repr-j', 'children'),
            Output('slice-repr-k', 'children'),
            Output('grid-points', 'points'),
            Output("fig1", "figure"),
            Output("fig2", "figure"),
            Output("fig3", "figure"),
            Output("fig4", "figure")
        ],
        
        [
            Input('upload-seismic', 'filename'),
            Input('attri-component', 'value'),
            Input('kernel-component', 'value'),
            Input('noise-component', 'value'),
            Input('threshold', 'value'),
            Input('scale-component', 'value'),
            Input("btn-a", "n_clicks"), 
            Input("btn-b", "n_clicks"),
        ]
)
def load_display_2d_3d_seismic(file, 
                               attribute, 
                               kernel, 
                               noise, 
                               threshold, 
                               cmap, 
                               prev, 
                               next):

    if file is None:
        PreventUpdate()

    volume_state, grid_points, volume = parse_seismic(file)

    ori_image, noise_red, attr = attributes(data=volume,
                                            attri_type=attribute,
                                            kernel=eval(kernel),
                                            noise=noise
                                            )
    mask = extMask(cube=attr, threshold=threshold)

    #next- prev
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if prev != 0 or next !=0:
        if 'btn-a' in changed_id:
            i = prev
        if 'btn-b' in changed_id:
            i = next
        if 'btn-b' not in changed_id:
            i = 0
            
    im1, im2  = ori_image[:,:,i+5], noise_red[:,:,i+5].compute() 
    im3, im4 = attr[:,:,i+5].compute(), mask[:, :, i+5].compute()

    return volume_state, volume_state, volume_state, grid_points, \
            plot(im1, cmap='gray', title='      Original Seismic Image'),\
            plot(im2, cmap='gray', title=f'     Denoised Seismic Image\n{noise}'),\
            plot(im3, cmap=cmap, title=f'       Seismic Attribute\n{attribute}'),\
            plot(im4, cmap='gray', title=f'     Seismic Mask')
        
    
@app.callback(
    [
        Output("slice-repr-i", "actor"), 
        Output("slice-repr-j", "actor"), 
        Output("slice-repr-k", "actor"), 
        Output("grid", "showCubeAxes"),
        Output("slice-repr-i", "property"),
        Output("slice-repr-i", "iSlice"),
        Output("slice-repr-j", "property"),
        Output("slice-repr-j", "jSlice"),
        Output("slice-repr-k", "property"),
        Output("slice-repr-k", "kSlice"),
        Output("slice-view", "triggerRender"),
    ],
    [
        Input("slider-i", "value"),
        Input("slider-j", "value"),
        Input("slider-k", "value"),
        Input("slider-lvl", "value"),
        Input("slider-window", "value"),
        Input('enabled', 'value'),

    ],
)
def update_seismic_slice_property(i, j, k, level, window, seis):

    render_call = random.random()

    if "inline" in seis:
        act0 = {'visibility':1}
    elif "inline" not in seis:
        act0 = {'visibility':0}
        
    if "xline" in seis:
        act1 = {'visibility':1}
    elif "xline" not in seis:
        act1 = {'visibility':0}
        
    if "z" in seis:
        act2 = {'visibility':1}
    elif "z" not in seis:
        act2 = {'visibility':0}
    
    if "grid" in seis:
        act3 = True
    elif "grid" not in seis:
        act3 = False

    slice_prop = {"colorLevel": level, "colorWindow": window}

    return act0, act1, act2, act3, slice_prop, i, slice_prop, j, slice_prop, k, render_call

# @app.callback(
#     Output('download_segy', 'data'),
#     Input('btn_segy', 'n_clicks'),
#     prevent_initial_call=True
# )
# def download_mask_volume(data):

#     data = numpy2segy(data)
#     return data

if __name__ == "__main__":
    app.run_server(debug=True)