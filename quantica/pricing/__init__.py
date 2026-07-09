"""Pricing subpackage: instruments, processes, and numerical engines.

The design follows the Instrument / Process / Engine separation (CLAUDE.md §4):
an *instrument* is the contract, a *process* is the market dynamics, and an
*engine* is a numerical method that prices an instrument under a process.
"""
