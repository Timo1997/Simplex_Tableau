
import pandas as pd
import numpy as np
import copy
import sympy as sp
from sympy import sympify


def get_pivotzeile(copy_tableau, pivot_spalte, anzahl_zeilen):
    # soll original Tableau nicht ändern
    copy_tableau = copy.deepcopy(copy_tableau)
    # wähle Ressourcenverbrauchskoeffizienten der Pivotspalte
    pivot_spalte_values =  copy_tableau.iloc[copy_tableau.index.difference([0, 1,  (anzahl_zeilen-1), (anzahl_zeilen-2)]), pivot_spalte]
    # wähle Menge der Restriktionen
    quantity = copy_tableau.iloc[copy_tableau.index.difference([0, 1, (anzahl_zeilen-1), (anzahl_zeilen-2)]), 2]
    #verhinden von teilen durch negative Zahlen und 0
    pivot_spalte_values.mask(pivot_spalte_values <= 0 , np.nan, inplace = True)
    #Hilfsmatrix zum ermitteln der Pivotspalte
    copy_tableau =  quantity / pivot_spalte_values
    #übergabe der Zeilenid mit dem kleinsten Wert
    return copy_tableau.astype(float).idxmin(skipna=True)

    
    
def get_pivotspalte(copy_tableau, infinite):
    # soll original Tableau nicht ändern
    copy_tableau = copy.deepcopy(copy_tableau)
    #Schleife über alle Spalten
    for column in copy_tableau:
        #nur Zeilen mit Ressourcenverbrauchskoeffizienten werden angesehen
        if column != 0 and column != 1 and column != 2:
            #zum Berechnen der größten cj-zj Zeile muss wenn nötig M durch ansatzweise unendlich ersetzt werden
            if isinstance(copy_tableau.iloc[-1,column], sp.Basic):  # Filtern der Felder mit M
                copy_tableau.iloc[-1,column] = copy_tableau.iloc[-1,column].subs(infinite, 9999)
            copy_tableau.iloc[-1,column] = int(copy_tableau.iloc[-1,column])
    #bestimmen des Spaltenid, welche den größten Wert enthält
    pivot_spalte = copy_tableau.iloc[-1,3:].astype(float).idxmax(axis=0)
    return pivot_spalte

#-----------------------------------------------------------------------------

def update_simplex_tableau(copy_tableau, pivot_zeile, pivot_spalte, anzahl_zeilen):
    #Pivotelelement wird auf Wert 1 gebracht indem man die Zeile durch das Pivotelement teilt
    copy_tableau.iloc[pivot_zeile, 2:] = (copy_tableau.iloc[pivot_zeile, 2:] / copy_tableau.iloc[pivot_zeile,pivot_spalte])
    #neue Basisvariable wird durch alte getauscht
    copy_tableau = update_pivotzeile(copy_tableau, pivot_zeile, pivot_spalte)
    #aktualisiere die restlichen Restritkionsmengen und die Ressourenverbrauchskoeffizienten
    copy_tableau = update_basis_variables(copy_tableau, pivot_zeile, pivot_spalte, anzahl_zeilen)
    return copy_tableau

def update_pivotzeile(copy_tableau, alte_basis_var, neue_basis_var):
    #aktualisiere den cj Wert der neuen Basisvariable
    copy_tableau.iloc[alte_basis_var, 0] = copy_tableau.iloc[0, neue_basis_var] 
    #aktualisiere den Namen der neuen Basisvariable
    copy_tableau.iloc[alte_basis_var, 1] = copy_tableau.iloc[1, neue_basis_var]
    return copy_tableau

def update_basis_variables(copy_tableau, pivot_zeile, pivot_spalte, anzahl_zeilen): 
    for index in copy_tableau.index:
        #wähle jede Zeile der gleich bleibenden Basisvariablen und bringen die Pivotspalte auf 0
        if index != pivot_zeile  and index != 0 and index != 1 and index != anzahl_zeilen-1 and index != anzahl_zeilen-2:      
            copy_tableau.iloc[index, copy_tableau.columns.difference([0, 1], sort=False)] = copy_tableau.iloc[index, copy_tableau.columns.difference([0,1], sort=False)] - ((copy_tableau.iloc[pivot_zeile, copy_tableau.columns.difference([0, 1], sort=False)] * copy_tableau.iloc[index, pivot_spalte]))         
            
    return copy_tableau

#----------------------------------------------------------------------------
def get_cj_zj(copy_tableau):
    #print(anzahl_zeilen)
    anzahl_zeilen = len(copy_tableau.index)
    #berechne Zeile zj 
    for column in range(0, len(copy_tableau.columns)):
        if column != 0 and column != 1:
            cj_basisvar = copy_tableau.iloc[copy_tableau.index.difference([0,1, anzahl_zeilen-1, anzahl_zeilen-2], sort=False ), 0]
            restr_var = copy_tableau.iloc[copy_tableau.index.difference([0,1, anzahl_zeilen-1, anzahl_zeilen-2], sort=False ), column] 
            temp = cj_basisvar * restr_var
            copy_tableau.iloc[-2, column] = temp.sum()
           
                
            
    
    #berechne Zeile cj-zj
    copy_tableau.iloc[-1, copy_tableau.columns.difference([0, 1, 2], sort=False )] = copy_tableau.iloc[0, copy_tableau.columns.difference([0 ,1 ,2], sort=False )] -  copy_tableau.iloc[-2, copy_tableau.columns.difference([0, 1,2], sort=False )]
    return copy_tableau

#Berechne maximalen cj-zj Wert
def get_max_cj_zj(copy_tableau, infinite):
    copy_tableau = copy.deepcopy(copy_tableau)
    for column in copy_tableau:
        if column != 0 and column != 1 and column != 2:
            if isinstance(copy_tableau.iloc[-1,column], sp.Expr):
                copy_tableau.iloc[-1,column] = copy_tableau.iloc[-1,column].subs(infinite, 9999)
            copy_tableau.iloc[-1,column] = int(copy_tableau.iloc[-1,column])
    max_value = copy_tableau.iloc[-1,3:].astype(float).max(axis=0)
    return max_value


#Prüfe auf Ausführbarkeit 
def check_infeasibility(last_tableau, liste_meldungen, finished):
    #Wenn in der finalen Lösungsmenge ein M ist, ist auch eine künstliche Variable in der Lösung
    #prüfe ob M vorhanden ist und ob eine Lösung gefunden wurde
    if isinstance(last_tableau.iloc[-2,2], sp.Basic) and finished:  
        liste_meldungen.append("Spezialfall: Unausführbarkeit (Infeasibility) -> Falls ein optimales Tableau eine künstliche Variable enthält, ist das Problem unlösbar („infeasible“).")

        
#Prüfe auf unbeschraenkten Lösungsraum
def check_unbeschraenkter_loesungsraum(check, liste_meldungen):
    #Wenn die Pivotzeile keine Zahl enthält wurde konnte kein Wert berechnet werden
    if np.isnan(check):
        liste_meldungen.append("Spezialfall: Unbeschränkter Lösungsraum -> keine zulässige Pivotzeile => Lösungsraum unbeschränkt.")
        return True
    else:
        return False
    
def simplex_algorithm(tableau, counter_limit, infinite):
    anzahl_zeilen = len(tableau.index)
    counter = 0 #Zähler für die Anzahl an Iterationen bis abgebrochen wird
    ende = False #Überprüfung ob der Simplex ein Ergebnis gefunden hat
    Meldungen = [] # Liste für die Fehlermeldung wird erzeugt
    list_pivot_elements = []
    list_tableaus = [copy.deepcopy(tableau.fillna(''))] # Anfangstableau wird in eine liste kopiert
    
    #Solange cj-zj noch einen positiven Wert hat, wird der Simplex Algorithmus ausgeführt
    while get_max_cj_zj(tableau, infinite) > 0 :
        Meldungen.append([]) #erzeuge eine Liste für Meldunge (bezieht sich auf vorheriges Tableau)
        Pivotspalte = get_pivotspalte(tableau, infinite)
        Pivotzeile = get_pivotzeile(tableau, Pivotspalte, anzahl_zeilen)
        list_pivot_elements.append([Pivotzeile, Pivotspalte])
        if check_unbeschraenkter_loesungsraum(Pivotzeile, Meldungen[counter]):
            #wenn der Lösungsraum unbeschränkt ist, wird abgebrochen
            break
        
        update_simplex_tableau(tableau, Pivotzeile, Pivotspalte, anzahl_zeilen)
        tableau = get_cj_zj(tableau)

        tableau = tableau.fillna('') #alle unnötigen Felder werden geleert
        list_tableaus.append(copy.deepcopy(tableau)) #füge das neue Tableau wieder in die Liste hinzu

        counter += 1
        if counter == counter_limit:
            break

    if get_max_cj_zj(tableau, infinite) <= 0:
                #Überprüfung ob ein Ergebnis gefunden wurde
                ende = True

    #Meldungen für das letzte Tableau 
    Meldungen.append([])
    list_pivot_elements.append([None,None])

    # kontrolliere Lösbarkeit
    check_infeasibility(list_tableaus[-1], Meldungen[-1], ende )
    
    return list_tableaus, Meldungen, list_pivot_elements
    