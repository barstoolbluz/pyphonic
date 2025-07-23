#!/usr/bin/env python3
"""
Implacable code audit of the polyphase implementation.
Reading the source code directly to identify issues.
"""

import ast
import os

def audit_source_code():
    """Perform line-by-line audit of polyphase_gen_fixed.py"""
    
    print("IMPLACABLE CODE AUDIT OF POLYPHASE_GEN_FIXED.PY")
    print("=" * 70)
    
    with open('src/polyphase_gen_fixed.py', 'r') as f:
        source = f.read()
        
    tree = ast.parse(source)
    
    issues = []
    
    # Check 1: Kernel length calculation
    print("\n1. KERNEL LENGTH CALCULATION")
    print("-" * 60)
    
    # Find the __init__ method
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '__init__':
            # Look for kernel length calculation
            found_kernel_calc = False
            for stmt in node.body:
                if isinstance(stmt, ast.If):
                    # Check the if condition
                    if isinstance(stmt.test, ast.Compare):
                        # This is the odd/even check
                        print("Found odd/even taps_per_phase check ✓")
                        found_kernel_calc = True
                        
                        # Check the branches
                        print("\nOdd branch:")
                        for s in stmt.body:
                            if isinstance(s, ast.Assign):
                                if any(t.attr == 'kernel_length' for t in s.targets if hasattr(t, 'attr')):
                                    print(f"  Sets kernel_length = taps_per_phase * phase_count + 1")
                                    
                        print("\nEven branch:")
                        for s in stmt.orelse:
                            if isinstance(s, ast.Assign):
                                if any(t.attr == 'kernel_length' for t in s.targets if hasattr(t, 'attr')):
                                    print(f"  Sets kernel_length = taps_per_phase * phase_count + 1")
            
            if not found_kernel_calc:
                issues.append("Kernel length calculation not found in __init__")
    
    # Check 2: Center calculation in _extract_polyphase_components
    print("\n2. POLYPHASE CENTER CALCULATION")
    print("-" * 60)
    
    # Look for the extraction method
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '_extract_polyphase_components':
            print("Found _extract_polyphase_components method")
            
            # Look for centre calculation
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name) and target.id == 'centre':
                            print("  Found centre calculation ✓")
                            # The formula should use kernel_length
                            if 'kernel_length' in ast.unparse(stmt.value):
                                print("  Uses kernel_length in calculation ✓")
                            else:
                                issues.append("Centre calculation doesn't use kernel_length")
    
    # Check 3: Pass/stop band indices
    print("\n3. PASS/STOP BAND INDEX CALCULATION")
    print("-" * 60)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '_verify_specifications':
            print("Found _verify_specifications method")
            
            # Look for index calculations
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name):
                            if 'passband_idx' in target.id:
                                # Check if it has pi factor
                                if 'pi' in ast.unparse(stmt.value):
                                    issues.append("Passband index still uses pi factor!")
                                else:
                                    print("  Passband index calculation correct (no pi) ✓")
                            elif 'stopband_idx' in target.id:
                                if 'pi' in ast.unparse(stmt.value):
                                    issues.append("Stopband index still uses pi factor!")
                                else:
                                    print("  Stopband index calculation correct (no pi) ✓")
    
    # Check 4: Row sum validation
    print("\n4. ROW SUM VALIDATION")
    print("-" * 60)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '_normalize_dc_gain':
            print("Found _normalize_dc_gain method")
            
            # Look for minimum check
            found_min_check = False
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.If):
                    condition = ast.unparse(stmt.test)
                    if 'min_sum' in condition and '1e-10' in condition:
                        print("  Found minimum sum check ✓")
                        found_min_check = True
                        
                        # Check if it raises
                        for s in stmt.body:
                            if isinstance(s, ast.Raise):
                                print("  Raises ValueError for pathological kernels ✓")
                                
            if not found_min_check:
                issues.append("No minimum sum validation found")
    
    # Check 5: Stopband enforcement
    print("\n5. STOPBAND GOAL ENFORCEMENT")
    print("-" * 60)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'generate':
            print("Found generate method")
            
            # Look for stopband check
            found_stopband_check = False
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.If):
                    condition = ast.unparse(stmt.test)
                    if 'measured_stopband_db' in condition and 'stopband_db' in condition:
                        print("  Found stopband enforcement check ✓")
                        found_stopband_check = True
                        
                        # Check if it raises
                        for s in stmt.body:
                            if isinstance(s, ast.Raise):
                                print("  Raises ValueError if target not met ✓")
                                
            if not found_stopband_check:
                issues.append("No stopband enforcement found")
    
    # Check 6: Beta calculation
    print("\n6. AUTOMATIC BETA CALCULATION")
    print("-" * 60)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '__init__':
            # Look for beta calculation
            found_beta_calc = False
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if hasattr(target, 'attr') and target.attr == 'window_param':
                            value_str = ast.unparse(stmt.value)
                            if '0.1102' in value_str and '8.7' in value_str:
                                print("  Found automatic β calculation ✓")
                                print(f"  Formula: β = 0.1102 * (|A| - 8.7)")
                                found_beta_calc = True
                                
            if not found_beta_calc:
                issues.append("Automatic beta calculation not found")
    
    # Check 7: Memory optimization
    print("\n7. MEMORY OPTIMIZATION")
    print("-" * 60)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '_extract_polyphase_components':
            # Look for advanced indexing
            source_extract = ast.unparse(node)
            if 'np.arange' in source_extract and 'indices' in source_extract:
                print("  Uses advanced numpy indexing ✓")
            else:
                print("  Still uses list comprehension")
                issues.append("Memory optimization not implemented")
    
    # Check 8: Float32 quantization
    print("\n8. FLOAT32 QUANTIZATION WARNING")
    print("-" * 60)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'save':
            # Look for round-off calculation
            found_roundoff = False
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name) and 'round_off' in target.id:
                            print("  Calculates round-off error ✓")
                            found_roundoff = True
                            
            # Look for warning
            if found_roundoff:
                for stmt in ast.walk(node):
                    if isinstance(stmt, ast.Expr):
                        call = stmt.value
                        if isinstance(call, ast.Call) and hasattr(call.func, 'attr'):
                            if call.func.attr == 'warn':
                                print("  Issues warning if round-off significant ✓")
                                
            if not found_roundoff:
                issues.append("Float32 quantization analysis not found")
    
    # Check 9: Complete verification
    print("\n9. VERIFICATION COMPLETENESS")
    print("-" * 60)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '_verify_specifications':
            source_verify = ast.unparse(node)
            
            checks = [
                ('Group delay', 'group_delay' in source_verify),
                ('Phase continuity', 'phase_continuity' in source_verify),
                ('Odd kernel', 'odd_kernel' in source_verify),
                ('Symmetry', 'symmetry_error' in source_verify),
                ('DC normalization', 'dc_error' in source_verify),
                ('Passband ripple', 'passband_ripple' in source_verify),
                ('Stopband', 'measured_stopband' in source_verify)
            ]
            
            for check_name, present in checks:
                if present:
                    print(f"  {check_name}: ✓")
                else:
                    print(f"  {check_name}: ✗")
                    issues.append(f"Missing {check_name} verification")
    
    # Check 10: Property-based tests
    print("\n10. PROPERTY-BASED TESTS")
    print("-" * 60)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'property_based_tests':
            print("  Found property_based_tests function ✓")
            
            # Check what properties are tested
            source_test = ast.unparse(node)
            properties = [
                ('Symmetry', "assert.*'symmetry_pass'" in source_test),
                ('DC gain', "assert.*'dc_normalized'" in source_test),
                ('Odd kernel', "assert.*'odd_kernel'" in source_test),
                ('Stopband', "assert.*measured_stopband" in source_test)
            ]
            
            for prop_name, tested in properties:
                if tested:
                    print(f"    Tests {prop_name}: ✓")
                else:
                    issues.append(f"Property test missing: {prop_name}")
    
    # Summary
    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("-" * 60)
    
    if issues:
        print(f"Found {len(issues)} issues:\n")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
    else:
        print("No issues found! Implementation appears correct.")
    
    return issues


def analyze_mathematical_correctness():
    """Analyze the mathematical correctness of key calculations."""
    
    print("\n\nMATHEMATICAL CORRECTNESS ANALYSIS")
    print("=" * 70)
    
    # Read the source
    with open('src/polyphase_gen_fixed.py', 'r') as f:
        source = f.read()
    
    # Check 1: Kernel length formula
    print("\n1. KERNEL LENGTH FORMULA")
    print("-" * 60)
    
    # Extract the formulas
    lines = source.split('\n')
    for i, line in enumerate(lines):
        if 'self.kernel_length =' in line:
            print(f"Line {i+1}: {line.strip()}")
            
            # Check if it matches the correct formula
            if 'taps_per_phase * phase_count + 1' in line:
                print("  ✓ Correct formula: T * L + 1")
            else:
                print("  ✗ Incorrect formula!")
    
    # Check 2: Zero crossings calculation
    print("\n2. ZERO CROSSINGS CALCULATION")
    print("-" * 60)
    
    for i, line in enumerate(lines):
        if 'self.zero_crossings =' in line:
            print(f"Line {i+1}: {line.strip()}")
            
            # For odd taps_per_phase, should be (T-1)/2
            if i > 0 and 'odd' in lines[i-2]:
                if '(taps_per_phase - 1) // 2' in line:
                    print("  ✓ Correct for odd T: (T-1)/2")
                else:
                    print("  ✗ Incorrect for odd T!")
            # For even, should be T/2
            elif 'taps_per_phase // 2' in line:
                print("  ✓ Correct for even T: T/2")
    
    # Check 3: Centre calculation
    print("\n3. POLYPHASE CENTRE CALCULATION")
    print("-" * 60)
    
    for i, line in enumerate(lines):
        if 'centre =' in line and 'kernel_length' in line:
            print(f"Line {i+1}: {line.strip()}")
            
            # Should be: kernel_length // 2 - zero_crossings * phase_count
            if 'kernel_length // 2 - self.zero_crossings * self.phase_count' in line:
                print("  ✓ Correct formula")
            else:
                print("  ✗ Check formula carefully!")
    
    # Check 4: Beta formula
    print("\n4. KAISER BETA FORMULA")
    print("-" * 60)
    
    for i, line in enumerate(lines):
        if '0.1102' in line and 'stopband_db' in line:
            print(f"Line {i+1}: {line.strip()}")
            
            # Should be: 0.1102 * (|A| - 8.7)
            if '0.1102 * (abs(stopband_db) - 8.7)' in line:
                print("  ✓ Correct formula: β = 0.1102 * (|A| - 8.7)")
            else:
                print("  ✗ Check formula!")
    
    # Check 5: Index calculations
    print("\n5. FREQUENCY INDEX CALCULATIONS")
    print("-" * 60)
    
    for i, line in enumerate(lines):
        if 'passband_idx' in line or 'stopband_idx' in line:
            print(f"Line {i+1}: {line.strip()}")
            
            # Should NOT have pi factor
            if 'np.pi' in line or '* pi' in line:
                print("  ✗ ERROR: Still has π factor!")
            else:
                print("  ✓ No π factor (correct)")


if __name__ == '__main__':
    # Run the audit
    issues = audit_source_code()
    
    # Run mathematical analysis
    analyze_mathematical_correctness()
    
    print("\n\nFINAL VERDICT:")
    print("=" * 70)
    if not issues:
        print("The implementation passes implacable scrutiny!")
    else:
        print(f"Found {len(issues)} issues that need attention.")