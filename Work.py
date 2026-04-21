import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import argparse

def main (args):
    # Parámetros del sistema
    resolution = 20 # pixels/um
    n = 2 # constante del dielectrico SiN
    n0 = args.n # constante del dielectrico relleno
    l = 1.55 #longitud de onda (um)
    w = args.w # width de la guía de onda (um)
    a = args.a # altura de la guía de onda (um)
    g = args.g # grosor del anillo (um)
    r = args.r # radio interno del anillo (um)
    d = args.d # distancia entre estructuras (um)
    pad = args.pad # espacio vacío (um)
    dpml = args.dpml # tamaño de límite (um)

    sx = 2*(r + w + g + pad + dpml)
    sy = 2*(r + w + g + pad + dpml) # Tamaño de la celda en y incluye el grosor de la guía y distancia entre guía y anillo (si hay)
    
    cell = mp.Vector3(sx,sy,0)
    
    # Geometrías (Anillo y guía)
    c1 = mp.Cylinder(radius=r + g, height=a, material=mp.Medium(index=n))
    c2 = mp.Cylinder(radius=r, height=a, material=mp.Medium(index=n0))

    blk = mp.Block(
            size=mp.Vector3(sx,w,a),
            center=mp.Vector3(0,-r-g-0.5*w-d), 
            material=mp.Medium(index=n)
            )

    # Fuente
    fcen = 1/l # frecuencia central (c/um)
    df = args.df

    src = [
        mp.EigenModeSource(
            src=mp.GaussianSource(fcen, fwidth=df),
            center=mp.Vector3(-sx/2+dpml+2,-r-g-0.5*w-d),
            size=mp.Vector3(0,w),
            direction=mp.X,
            eig_band=1,
            eig_parity=mp.EVEN_Y,
            eig_match_freq=True,
            eig_kpoint=mp.Vector3(1,0,0),
        )
    ]

    #Sensores
    freg_in = mp.FluxRegion(
        center=mp.Vector3(-sx/2+dpml+3,-r-g-0.5*w-d), # Posición del sensor del input, 1um a la derecha de la fuente 
        size=mp.Vector3(0, 5*w) # Tamaño del plano de medición, lo hacemos un poco más grande que el ancho de la guía para asegurarnos de capturar todo el flujo
    )

    freg_out = mp.FluxRegion(
        center=mp.Vector3(sx/2-dpml-2,-r-g-0.5*w-d), # Posición del plano donde se mide el flujo, justo al final de la guía
        size=mp.Vector3(0, 5*w) # Tamaño del plano de medición, lo hacemos un poco más grande que el ancho de la guía para asegurarnos de capturar todo el flujo
    ) 
    
    nfreq = 500 # número de frecuencias que computar

    # Simulación

    # Primero simulamos el que será nuestro sistema de referencia, el bloque sin anillo
    sim = mp.Simulation(
        cell_size=cell,     
        geometry=[blk],
        boundary_layers=[mp.PML(dpml)],
        sources=src,
        resolution=resolution,
        default_material=mp.Medium(index=n0)
    )

    # Computación del flujo
    trans_ref_in = sim.add_flux(fcen,df,nfreq,freg_in)
    trans_ref_out = sim.add_flux(fcen,df,nfreq,freg_out)

    # Simulación se hace y nosotros medimos cuando la fuente acaba ya
    sim.run(until_after_sources=mp.stop_when_fields_decayed(150, mp.Ez, mp.Vector3(0,-r-g-0.5*w-d), args.ld)) # Después de que la fuente se apague, se sigue simulando hasta que el campo Ez en la posición dada decaiga a menos de 1e-6 veces su valor máximo, o hasta un máximo de 200 unidades de tiempo

    # Obtener datos
    flux_ref_in = np.array(mp.get_fluxes(trans_ref_in)) # Obtiene los valores de flujo para cada frecuencia medida en el plano definido por freg. En un array para poder hacer los posteriores cálculos y gráficos.
    flux_ref_out = np.array(mp.get_fluxes(trans_ref_out)) # Obtiene los valores de flujo para cada frecuencia medida en el plano definido por freg. En un array para poder hacer los posteriores cálculos y gráficos.
    freqs_ref = np.array(mp.get_flux_freqs(trans_ref_out)) # Obtiene las frecuencias correspondientes a cada valor de flujo, basándose en la frecuencia central y el grosor definidos al crear el objeto de flujo
    
    # Normalización del flujo 1
    I_ref = np.divide(flux_ref_out, flux_ref_in)

    # Reseteamos la simulación para el sistema con anillo
    sim.reset_meep()
    
    # Simulación del sistema con anillo
    sim = mp.Simulation(
        cell_size=cell,
        geometry=[c1,c2,blk],
        boundary_layers=[mp.PML(dpml)],
        sources=src,
        resolution=resolution,
        default_material=mp.Medium(index=n0)
    )

    # Computación del flujo
    trans_in = sim.add_flux(fcen,df,nfreq,freg_in)
    trans_out = sim.add_flux(fcen,df,nfreq,freg_out)

    # Simulación se hace y nosotros medimos cuando la fuente acaba ya
    sim.run(until_after_sources=mp.stop_when_fields_decayed(150, mp.Ez, mp.Vector3(0,-r-g-0.5*w-d), args.ld)) # Después de que la fuente se apague, se sigue simulando hasta que el campo Ez en la posición dada decaiga a menos de 1e-6 veces su valor máximo, o hasta un máximo de 200 unidades de tiempo

    # Obtener datos
    flux_in = np.array(mp.get_fluxes(trans_in)) # Obtiene los valores de flujo para cada frecuencia medida en el plano definido por freg. En un array para poder hacer los posteriores cálculos y gráficos.
    flux_out = np.array(mp.get_fluxes(trans_out)) # Obtiene los valores de flujo para cada frecuencia medida en el plano definido por freg. En un array para poder hacer los posteriores cálculos y gráficos.
    freqs = np.array(mp.get_flux_freqs(trans_out)) # Obtiene las frecuencias correspondientes a cada valor de flujo, basándose en la frecuencia central y el grosor definidos al crear el objeto de flujo

    # Normalización del flujo 2
    I_ring = np.divide(flux_out, flux_in)

    # Normalización del flujo 3
    I = np.divide(flux_out, flux_ref_out)

    # Plots
    # Señal de entrada
    plt.plot(freqs_ref, flux_ref_in, 'b-', label='Referencia')
    plt.plot(freqs, flux_in, 'r-', label='Con anillo')
    plt.xlabel("Frecuencia (c/um)")
    plt.ylabel("Flujo")
    plt.title("Flujo de entrada")
    plt.legend()
    plt.show()

    # Señal de salida
    plt.plot(freqs_ref, flux_ref_out, 'b-', label='Referencia')
    plt.plot(freqs, flux_out, 'r-', label='Con anillo')
    plt.xlabel("Frecuencia (c/um)")
    plt.ylabel("Flujo")
    plt.title("Flujo de salida")
    plt.legend()
    plt.show()

    # Intensidades
    # Caso 1
    plt.plot(freqs_ref, I_ref)
    plt.xlabel("Frecuencia (c/um)")
    plt.ylabel("Intensidad")
    plt.title("Espectro de la guía")
    plt.show()
    
    # Caso 2
    plt.plot(freqs, I_ring)
    plt.xlabel("Frecuencia (c/um)")
    plt.ylabel("Intensidad")
    plt.title("Espectro del anillo resonador independiente de la guía")
    plt.show()

    # Caso 3
    plt.plot(freqs, I)
    plt.xlabel("Frecuencia (c/um)")
    plt.ylabel("Intensidad")
    plt.title("Espectro del anillo resonador Normalizado a la guía")
    plt.show()

# Specify command lines for argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=float, default=1.5, help='Índice de refracción del material de relleno del sistema')
    parser.add_argument('-w', type=float, default=1, help='Ancho de la guía de onda')
    parser.add_argument('-a', type=float, default=0.3, help='Altura de la guía de onda')
    parser.add_argument('-g', type=float, default=1, help='Grosor del anillo')
    parser.add_argument('-r', type=float, default=54, help='Radio interno del anillo')
    parser.add_argument('-d', type=float, default=0.1, help='Distancia Guía-Anillo')
    parser.add_argument('-pad', type=float, default=4, help='Distancia de Vacío')
    parser.add_argument('-dpml', type=float, default=2, help='Dsitancia de frontera')
    parser.add_argument('-fcen', type=float, default=0.1, help='Frecuencia central')
    parser.add_argument('-df', type=float, default=0.18, help='Grosor de frecuencia')
    parser.add_argument('-ld', type=float, default=1e-5, help='Límite de Decaida')
    args = parser.parse_args()
    main(args)    